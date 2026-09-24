"""Anomaly / salient-deviation detection test (2026-09-23/24, overnight
autonomous battery). A sequence of T=8 2D vectors is fed one per
timestep: 7 drawn i.i.d. from a fixed "normal" cluster (small Gaussian
around the origin), and exactly 1 (at a random, a-priori-unknown
position) drawn from a clearly different distribution (shifted mean,
larger magnitude). The decode readout at the final step must regress the
CONTINUOUS VALUE of the outlier vector itself (kept a regression target,
not a discrete position/class index, consistent with this project's
established finding that discrete-classification framings of a task fail
even when the underlying regression framing succeeds -- EXPERIMENTS.md
2026-09-22/23 octant-classification result).

Same sequence-unroll mechanism as the other new temporal tasks in this
batch, same substrate/null/stats protocol. n=8 seeds,
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
DIM = 2
N_ENCODE = 2
N_READOUT = 30
SUBGRAPH_SIZE = 3000
T_STEPS = 8
EPOCHS = 300
LR = 3e-3
SEEDS = list(range(8))
N_TRAIN = 200
N_TEST = 100
NORMAL_STD = 0.15
OUTLIER_MIN_MAG, OUTLIER_MAX_MAG = 1.0, 2.0


def sample_batch(n: int, rng: np.random.Generator):
    seq = rng.normal(0, NORMAL_STD, size=(n, T_STEPS, DIM)).astype(np.float32)
    outlier_pos = rng.integers(0, T_STEPS, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    mag = rng.uniform(OUTLIER_MIN_MAG, OUTLIER_MAX_MAG, size=n)
    outlier_val = np.stack([mag * np.cos(theta), mag * np.sin(theta)], axis=1).astype(np.float32)
    outlier_val += rng.normal(0, NORMAL_STD, size=(n, DIM)).astype(np.float32)
    for i in range(n):
        seq[i, outlier_pos[i], :] = outlier_val[i]
    return seq, outlier_val


def forward_sequence(brain: RateBrain, step_seq_t: torch.Tensor) -> torch.Tensor:
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


def train_sequence(brain: RateBrain, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float):
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    X_t = torch.as_tensor(X, device=DEVICE)
    Y_t = torch.as_tensor(Y, device=DEVICE)
    for epoch in range(epochs):
        opt.zero_grad()
        pred = forward_sequence(brain, X_t)
        loss = ((pred - Y_t) ** 2).mean()
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
    print(f"subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=T_STEPS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_anomaly_detection_multiseed.jsonl"
    results = {"real": [], "null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = sample_batch(N_TRAIN, np.random.default_rng(9500 + seed))
            X_test, Y_test = sample_batch(N_TEST, np.random.default_rng(19500 + seed))
            X_test_t = torch.as_tensor(X_test, device=DEVICE)
            Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed}
            for name, sub in [("real", sub_real), ("null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train_sequence(brain, X, Y, epochs=EPOCHS, lr=LR)
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

    print("\n=== SONUC (anomali/sapma tespiti, n=8) ===")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_anomaly_detection_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_anomaly_detection_summary.json")


if __name__ == "__main__":
    main()
