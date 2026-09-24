"""Chaotic time-series prediction test (2026-09-23/24, overnight
autonomous battery). Standard Mackey-Glass benchmark (tau=17, the classic
chaotic reservoir-computing test case), directly connecting to this
project's own cited literature (Costi et al. 2025, "FlyWire connectome as
a reservoir for chaotic time-series prediction" -- already in
EXPERIMENTS.md/the published article's references) rather than a
synthetic task invented for this project.

Design choice (documented, not hidden): each of the 8 seeds gets its own
independently-generated Mackey-Glass series (different initial condition),
split 70/30 train/test. Within a series, prediction is done via
INDEPENDENT sliding windows (hidden state reset at the start of each
window, 8 raw past values fed one-per-timestep through the recurrence,
predicting the next value) rather than one continuous teacher-forced
unroll across the whole series -- a standard windowed-regression
simplification, chosen for tractability (a ~1400-step single BPTT unroll
would be a very different, much slower, harder-to-debug experiment).

Same substrate/null-model/stats protocol as everywhere else. n=8 seeds,
degree_preserving_rewire null.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_ENCODE = 1  # one scalar value injected per timestep
N_READOUT = 30
SUBGRAPH_SIZE = 3000
WINDOW = 8  # past values used to predict the next one
EPOCHS = 300
LR = 3e-3
SEEDS = list(range(8))
SERIES_LEN = 2000
TAU = 17
BETA, GAMMA, N_EXP = 0.2, 0.1, 10
TRAIN_FRAC = 0.7


def generate_mackey_glass(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hist = TAU
    x = np.zeros(length + hist + 1)
    x[:hist] = 1.2 + 0.05 * rng.standard_normal(hist)
    for t in range(hist, length + hist):
        x_tau = x[t - TAU]
        x[t + 1] = x[t] + (BETA * x_tau / (1 + x_tau ** N_EXP) - GAMMA * x[t])
    series = x[hist:]
    return (series - series.mean()) / (series.std() + 1e-8)


def windows_from_series(series: np.ndarray):
    X, Y = [], []
    for i in range(len(series) - WINDOW):
        X.append(series[i:i + WINDOW])
        Y.append(series[i + WINDOW])
    return np.asarray(X, dtype=np.float32)[:, :, None], np.asarray(Y, dtype=np.float32)[:, None]


def forward_sequence(brain: RateBrain, step_seq_t: torch.Tensor) -> torch.Tensor:
    """step_seq_t: (batch, T, N_ENCODE). Hidden state reset each call
    (independent-window design, see module docstring)."""
    cfg = brain.config
    batch, T, _ = step_seq_t.shape
    w = brain.effective_weights()
    a = torch.sigmoid(brain.leak_logit).unsqueeze(1)
    bias = brain.bias.unsqueeze(1)
    h = torch.zeros(brain.n, batch, device=step_seq_t.device, dtype=step_seq_t.dtype)
    for t in range(T):
        inject = torch.zeros(brain.n, batch, device=step_seq_t.device, dtype=step_seq_t.dtype)
        inject.index_add_(0, brain.encode_idx, step_seq_t[:, t, :].t())
        src = h.index_select(0, brain.col)
        messages = src * w.unsqueeze(1)
        pre = torch.zeros(brain.n, batch, device=step_seq_t.device, dtype=step_seq_t.dtype).index_add_(0, brain.row, messages)
        pre = pre + inject
        h = (1 - a) * h + a * torch.tanh(pre + bias)
    decode_h = h.index_select(0, brain.decode_idx)
    delta = brain.readout_W @ decode_h
    return delta.t() * cfg.decode_scale


def train_sequence(brain: RateBrain, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, batch_size: int = 256):
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    X_t = torch.as_tensor(X, device=DEVICE)
    Y_t = torch.as_tensor(Y, device=DEVICE)
    n = X_t.shape[0]
    gen = torch.Generator().manual_seed(0)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)[:batch_size]
        opt.zero_grad()
        pred = forward_sequence(brain, X_t[perm])
        loss = ((pred - Y_t[perm]) ** 2).mean()
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}/{epochs}  mse={loss.item():.4f}")


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_ENCODE, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, {len(encode_idx)} encode / {len(decode_idx)} decode")

    cfg = RateBrainConfig(dim=1, n_readout=len(decode_idx), T=WINDOW, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_mackey_glass_prediction_multiseed.jsonl"
    results = {"real": [], "null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            series = generate_mackey_glass(SERIES_LEN, seed=30000 + seed)
            split = int(len(series) * TRAIN_FRAC)
            X_all, Y_all = windows_from_series(series)
            n_train = split - WINDOW
            X_train, Y_train = X_all[:n_train], Y_all[:n_train]
            X_test, Y_test = X_all[n_train:], Y_all[n_train:]
            X_test_t = torch.as_tensor(X_test, device=DEVICE)
            Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed, "n_train": len(X_train), "n_test": len(X_test)}
            for name, sub in [("real", sub_real), ("null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train_sequence(brain, X_train, Y_train, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    pred = forward_sequence(brain, X_test_t)
                    test_mse = float(((pred - Y_test_t) ** 2).mean().item())
                results[name].append(test_mse)
                row[f"{name}_test_mse"] = test_mse
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed}: {row}")

    real = np.array(results["real"])
    null = np.array(results["null"])
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33

    print("\n=== SONUC (Mackey-Glass kaotik zaman serisi tahmini, n=8) ===")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_mackey_glass_prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_mackey_glass_prediction_summary.json")


if __name__ == "__main__":
    main()
