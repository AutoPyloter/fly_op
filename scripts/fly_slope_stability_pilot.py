"""Real spatial task pilot: slope-stability critical-slip-surface search
(benchmarks_geo.py) instead of another synthetic Rastrigin variant --
requested 2026-09-17 ("gercek uzamsal bir gorev dene"), the user's own
domain (geotechnical engineering). Uses the same differentiable
leaky-rate substrate + connectivity-verified encode/decode selection
validated today (rate_brain.py, EXPERIMENTS.md 2026-09-17 rate-brain
pilot v2). Small scale first, single seed, informal -- not pre-registered.

No "offset trick" is possible here (no closed-form optimum), so the
supervision target is the LOCAL finite-difference descent direction
(fly_proposer_scene._rays / _teacher_delta) on each random problem
instance (random slope geometry + soil properties, random trial-circle
start), same as fly_swarm_social's/PSO's teacher, just a real physical
objective instead of Rastrigin.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import er_null
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
)

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 400
LR = 3e-3
RAY_RADIUS = 1.0  # meters -- ~5-10% of typical slope-height scale


def build_training_set_geo(n_problems: int, n_starts: int, seed: int):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = random_geometry(rng)
        lo, hi = geo.param_bounds()
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi)
            fx = f(x)
            rays = _rays(x, f, fx, RAY_RADIUS)
            target = _teacher_delta(x, rays, RAY_RADIUS)  # local descent direction, no known global optimum
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(DIM)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def train(brain: RateBrain, X, Y, epochs: int, lr: float):
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    X_t, Y_t = torch.as_tensor(X), torch.as_tensor(Y)
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


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    print(f"verified-connected: {len(encode_full)} encode, {len(decode_full)} decode neurons")

    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")
    sub_null = er_null(sub_real, seed=9000)

    X, Y = build_training_set_geo(n_problems=25, n_starts=12, seed=9500)
    print(f"training set: {X.shape[0]} samples (real slope-stability problems, {X.shape[1]} rays)")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    results = {}
    for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
        print(f"\n--- training on {name} ---")
        brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=42)
        curve = train(brain, X, Y, epochs=EPOCHS, lr=LR)
        results[name] = curve[-1]
        print(f"{name}: final train mse = {curve[-1]:.4f}")

    print("\n=== SONUC (sev stabilitesi, gercek uzamsal gorev) ===")
    print(f"real_connectome final mse: {results['real_connectome']:.4f}")
    print(f"er_null         final mse: {results['er_null']:.4f}")
    diff_pct = (results['er_null'] - results['real_connectome']) / results['er_null'] * 100
    print(f"fark: gercek connectome er_null'dan {diff_pct:+.1f}% farkli (pozitif = gercek daha iyi)")

    import json
    with open("C:/projeler/fly_op/results/fly_slope_stability_pilot.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "diff_pct": diff_pct, "n_samples": X.shape[0]}, f, indent=2)
    print("wrote results/fly_slope_stability_pilot.json")


if __name__ == "__main__":
    main()
