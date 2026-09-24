"""Differentiable leaky-rate substrate (2026-09-17, "kucuk olcekte once
dene kur" -- after reviewing cesp99/fly-chess, EXPERIMENTS.md 2026-09-17).
Not pre-registered, not a phase -- a pilot to test one hypothesis: were
FlyOpt's negative/null results (Faz 1/1b/1c/S1, PSO, synapse plasticity)
partly an artifact of the LIF spiking substrate being non-differentiable
(hard threshold, numpy round-trip every step), forcing every "training"
attempt into weak zeroth-order/black-box search?

Same discipline as everywhere else in this project: connection topology
and Dale's-law sign are FIXED, taken straight from the real graph (or a
null model of it); only magnitude (`gain`, via `w = sign * softplus(gain)`,
fly-chess's own parameterisation) plus bias/leak/readout are trainable.
Dynamics: `h_{t+1} = (1-a)*h_t + a*act(W@h_t + bias + injected)`, a
continuous leaky-rate unit instead of a spiking LIF -- fully
autograd-compatible, no surrogate gradient needed, because there is no
hard threshold anywhere in the recurrence.

Runs on an induced SUBGRAPH (a few thousand neurons around the
encode/decode pools, not the full 139k/15M graph) purely for pilot speed;
see `scripts/fly_rate_brain_pilot.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


@dataclass(frozen=True)
class RateBrainConfig:
    dim: int
    n_readout: int = 30
    T: int = 8
    ray_radius: float = 0.3
    decode_scale: float = 0.5
    train_gain: bool = True  # False = magnitudes frozen at their real/null init (readout-only, like Faz S1)


def build_subgraph(weights_csr: sparse.csr_matrix, must_include: np.ndarray, n_total: int, seed: int):
    """Induced subgraph on `n_total` neurons chosen uniformly at random.

    2026-09-17 finding: on a 3,000-node uniform sample, the encode neurons'
    real out-edges (each neuron has ~100 of ~139k possible targets) almost
    never land inside the sample -- out-degree 0 in the induced subgraph,
    so an injected signal has nowhere to go and training silently
    flatlines. Kept for reference; superseded by `build_subgraph_bfs`
    below for anything that actually needs the injected signal to
    propagate."""
    rng = np.random.default_rng(seed)
    n = weights_csr.shape[0]
    must = np.unique(must_include)
    remaining = np.setdiff1d(np.arange(n), must)
    extra = rng.choice(remaining, size=max(0, n_total - len(must)), replace=False)
    nodes = np.concatenate([must, extra])
    rng.shuffle(nodes)
    old_to_new = -np.ones(n, dtype=np.int64)
    old_to_new[nodes] = np.arange(len(nodes))

    sub = weights_csr[nodes][:, nodes].tocsr()
    new_must = old_to_new[must_include]  # preserves must_include's original order/duplicates
    return sub, new_must, nodes


def build_subgraph_bfs(weights_csr: sparse.csr_matrix, encode_idx: np.ndarray, decode_idx: np.ndarray,
                        n_total: int, seed: int):
    """Induced subgraph that GUARANTEES forward connectivity from
    `encode_idx`: BFS outward along real directed edges (pre -> post,
    `weights_csr[post, pre]` convention, same as graph_builders.py) from
    the encode neurons, breadth-first, until `n_total` nodes are collected
    (encode_idx and decode_idx always included). Falls back to adding
    random nodes only if the forward frontier runs dry before n_total is
    reached (a real dead end, reported, not hidden)."""
    rng = np.random.default_rng(seed)
    n = weights_csr.shape[0]
    coo = weights_csr.tocoo()
    # weights_csr[post, pre]: successors of `pre` are the rows where col == pre
    succ: dict[int, np.ndarray] = {}
    order = np.argsort(coo.col, kind="stable")
    col_sorted, row_sorted = coo.col[order], coo.row[order]
    starts = np.searchsorted(col_sorted, np.arange(n + 1))

    def successors(node: int) -> np.ndarray:
        if node not in succ:
            succ[node] = row_sorted[starts[node]:starts[node + 1]]
        return succ[node]

    included = set(np.unique(np.concatenate([encode_idx, decode_idx])).tolist())
    frontier = list(included)
    while len(included) < n_total and frontier:
        nxt = []
        for node in frontier:
            for s in successors(node):
                s = int(s)
                if s not in included:
                    included.add(s)
                    nxt.append(s)
                    if len(included) >= n_total:
                        break
            if len(included) >= n_total:
                break
        frontier = nxt

    if len(included) < n_total:
        remaining = np.setdiff1d(np.arange(n), np.array(list(included), dtype=np.int64))
        fill = rng.choice(remaining, size=n_total - len(included), replace=False)
        included.update(int(x) for x in fill)

    nodes = np.array(sorted(included), dtype=np.int64)[:n_total]
    old_to_new = -np.ones(n, dtype=np.int64)
    old_to_new[nodes] = np.arange(len(nodes))
    sub = weights_csr[nodes][:, nodes].tocsr()
    return sub, old_to_new[encode_idx], old_to_new[decode_idx], nodes


class RateBrain(nn.Module):
    def __init__(self, sub_weights_csr: sparse.csr_matrix, encode_idx: np.ndarray, decode_idx: np.ndarray,
                 config: RateBrainConfig, seed: int):
        super().__init__()
        self.config = config
        self.n = sub_weights_csr.shape[0]
        coo = sub_weights_csr.tocoo()
        self.register_buffer("row", torch.as_tensor(coo.row, dtype=torch.int64))
        self.register_buffer("col", torch.as_tensor(coo.col, dtype=torch.int64))
        sign = torch.as_tensor(np.sign(coo.data), dtype=torch.float32)
        self.register_buffer("sign", sign)
        self.register_buffer("encode_idx", torch.as_tensor(encode_idx, dtype=torch.int64))
        self.register_buffer("decode_idx", torch.as_tensor(decode_idx, dtype=torch.int64))

        magnitude = torch.as_tensor(np.abs(coo.data), dtype=torch.float32).clamp(min=1e-4)
        init_gain = magnitude + torch.log(-torch.expm1(-magnitude))  # inverse softplus
        if config.train_gain:
            self.gain = nn.Parameter(init_gain)
        else:
            self.register_buffer("gain", init_gain)

        gen = torch.Generator().manual_seed(seed)
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.leak_logit = nn.Parameter(torch.zeros(self.n))  # sigmoid(0) = 0.5, matches fly-chess's alpha init
        self.readout_W = nn.Parameter(torch.randn(config.dim, len(decode_idx), generator=gen) / np.sqrt(len(decode_idx)))

    def effective_weights(self) -> torch.Tensor:
        return self.sign * F.softplus(self.gain)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """rays: (batch, n_encode) -- injected as constant additive drive into
        encode_idx every step. Returns (batch, dim) proposed delta."""
        cfg = self.config
        batch = rays.shape[0]
        w = self.effective_weights()
        a = torch.sigmoid(self.leak_logit).unsqueeze(1)  # (n, 1)
        bias = self.bias.unsqueeze(1)  # (n, 1)
        h = torch.zeros(self.n, batch, device=rays.device, dtype=rays.dtype)
        inject = torch.zeros(self.n, batch, device=rays.device, dtype=rays.dtype)
        inject.index_add_(0, self.encode_idx, rays.t())

        for t in range(cfg.T):
            src = h.index_select(0, self.col)  # (nnz, batch)
            messages = src * w.unsqueeze(1)  # (nnz, batch)
            pre = torch.zeros(self.n, batch, device=rays.device, dtype=rays.dtype).index_add_(0, self.row, messages)
            pre = pre + inject
            h = (1 - a) * h + a * torch.tanh(pre + bias)

        decode_h = h.index_select(0, self.decode_idx)  # (n_readout, batch)
        delta = self.readout_W @ decode_h  # (dim, batch)
        return delta.t() * cfg.decode_scale


def select_connected_encode_decode(
    weights_csr: sparse.csr_matrix,
    afferent_pool: np.ndarray,
    efferent_pool: np.ndarray,
    n_encode: int,
    n_decode: int,
    max_hops: int,
    n_encode_candidates: int,
    seed: int,
):
    """Pick encode/decode neurons that are VERIFIED forward-connected on
    the FULL graph within `max_hops` -- fixes the 2026-09-17 finding that
    `rng.choice` from the afferent/efferent pools (used everywhere in this
    project's encode/decode selection, including this file's own first
    pilot) can land on a pair with zero real path between them, silently
    making every "ray"/injected signal undeliverable.

    Multi-hop BFS forward from `n_encode_candidates` random afferent-pool
    neurons on the FULL connectome (not a subgraph); keeps only the
    encode candidates that reach at least one efferent-pool neuron, and
    the efferent-pool neurons actually reached."""
    rng = np.random.default_rng(seed)
    n = weights_csr.shape[0]
    coo = weights_csr.tocoo()
    order = np.argsort(coo.col, kind="stable")
    col_sorted, row_sorted = coo.col[order], coo.row[order]
    starts = np.searchsorted(col_sorted, np.arange(n + 1))

    def successors(nodes: np.ndarray) -> np.ndarray:
        out = [row_sorted[starts[node]:starts[node + 1]] for node in nodes]
        return np.unique(np.concatenate(out)) if out else np.empty(0, dtype=np.int64)

    candidates = rng.choice(afferent_pool, size=n_encode_candidates, replace=False)
    efferent_set = set(efferent_pool.tolist())
    reach_per_candidate: dict[int, set[int]] = {int(c): set() for c in candidates}
    frontier_per_candidate = {int(c): np.array([c]) for c in candidates}

    for _hop in range(max_hops):
        for c in list(frontier_per_candidate):
            nxt = successors(frontier_per_candidate[c])
            hit = [int(x) for x in nxt if int(x) in efferent_set]
            reach_per_candidate[c].update(hit)
            frontier_per_candidate[c] = nxt

    ranked = sorted(reach_per_candidate.items(), key=lambda kv: -len(kv[1]))
    encode_idx = np.array([c for c, r in ranked[:n_encode] if len(r) > 0], dtype=np.int64)
    if len(encode_idx) == 0:
        raise RuntimeError("no encode candidate reached any efferent neuron within max_hops -- increase "
                            "max_hops or n_encode_candidates")
    reachable_decode = set()
    for c in encode_idx:
        reachable_decode.update(reach_per_candidate[int(c)])
    decode_idx = np.array(sorted(reachable_decode)[:n_decode], dtype=np.int64)
    return encode_idx, decode_idx


def build_training_set(dim: int, bounds: tuple[float, float], n_problems: int, n_starts: int, ray_radius: float, seed: int):
    """Same multi-problem/multi-start scheme used throughout 2026-09-17's
    exploration (fly_swarm_social.py, tiny_recursive_proposer.py) -- many
    randomly-offset Rastrigin instances, ideal direction towards the known
    true minimum, encoded as `2*dim` finite-difference rays."""
    from flyopt.benchmarks import rastrigin
    from flyopt.variants.fly_proposer_scene import _rays

    rng = np.random.default_rng(seed)
    lo, hi = bounds
    X, Y = [], []
    for _ in range(n_problems):
        offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
        f = lambda x, off=offset: rastrigin(x - off)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi, size=dim)
            fx = f(x)
            rays = _rays(x, f, fx, ray_radius)
            target = offset - x
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(dim)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def train(brain: RateBrain, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float) -> list[float]:
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    device = brain.sign.device
    X_t, Y_t = torch.as_tensor(X, device=device), torch.as_tensor(Y, device=device)
    curve = []
    for epoch in range(epochs):
        opt.zero_grad()
        pred = brain(X_t)
        loss = ((pred - Y_t) ** 2).mean()
        loss.backward()
        opt.step()
        curve.append(float(loss.item()))
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}/{epochs}  mse={loss.item():.4f}")
    return curve
