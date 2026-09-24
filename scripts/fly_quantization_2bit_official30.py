"""n=8 pilot (fly_quantization_robustness_multiseed.py) showed the 2-bit
quantization level crossing the Cliff's-delta magnitude threshold
(delta=0.469, |delta|>0.33) though not p<0.05 (p=0.130). Per standing
protocol (magnitude alone triggers an n=30 extension), this runs 22 new
seeds (8-29) at the 2-bit level only and combines with the original 0-7
for the official n=30 verdict. 8-bit and 4-bit pilots were clean
non-significant ties (delta=-0.28 and 0.00) and are not extended.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
    train,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
N_BITS = 2
SEEDS = list(range(8, 30))


def build_training_set_geo(n_problems, n_starts, seed):
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
            target = _teacher_delta(x, rays, RAY_RADIUS)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(DIM)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def eval_mse(brain, X_t, Y_t) -> float:
    with torch.no_grad():
        pred = brain(X_t)
        return float(((pred - Y_t) ** 2).mean().item())


def quantized_mse(brain, X_t, Y_t, n_bits: int) -> float:
    original_gain = brain.gain.data.clone()
    with torch.no_grad():
        w = brain.effective_weights()
        w_max = w.abs().max().clamp(min=1e-8)
        levels = 2 ** n_bits
        step = (2 * w_max) / levels
        w_q = torch.round(w / step) * step
        w_q = w_q.clamp(-w_max, w_max)
        mag_q = w_q.abs().clamp(min=1e-6)
        new_gain = mag_q + torch.log(-torch.expm1(-mag_q))
        brain.gain.data.copy_(new_gain)
        brain.sign.copy_(torch.sign(w_q) + (w_q == 0).float() * brain.sign)
        mse = eval_mse(brain, X_t, Y_t)
    brain.gain.data.copy_(original_gain)
    return mse


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_quantization_2bit_seeds8to29.jsonl"
    real_mses, null_mses = [], []
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            X_test_t = torch.as_tensor(X_test, device=DEVICE)
            Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed}
            for name, sub, store in [("real", sub_real, real_mses), ("null", sub_null, null_mses)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                q_mse = quantized_mse(brain, X_test_t, Y_test_t, N_BITS)
                store.append(q_mse)
                row[f"{name}_mse_2bit"] = q_mse
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed}: {row}")

    orig = [json.loads(l) for l in open("C:/projeler/fly_op/results/fly_quantization_robustness_multiseed.jsonl", encoding="utf-8")]
    real = np.array([r["real_mse_2bit"] for r in orig] + real_mses)
    null = np.array([r["null_mse_2bit"] for r in orig] + null_mses)
    assert len(real) == 30 and len(null) == 30

    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33

    print("\n=== SONUC (2-bit kuantizasyon, n=30) ===")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Mann-Whitney p={mw_p:.6f}  Wilcoxon p={w_p:.6f}  Cliff's delta={cliffs_delta:.3f}  gate={'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_quantization_2bit_official30_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_quantization_2bit_official30_summary.json")


if __name__ == "__main__":
    main()
