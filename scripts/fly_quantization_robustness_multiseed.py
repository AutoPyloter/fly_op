"""Weight-quantization robustness test (2026-09-24) -- relevant to the
"where is this actually used" question (neuromorphic/edge-hardware
deployment typically requires low-bit-precision weights). Trains real vs
null RateBrain on the official slope-stability task (identical recipe to
the official n=30 result). At inference time only (no retraining),
quantizes the EFFECTIVE recurrent weights to N-bit signed fixed-point
(uniform quantization over each network's own weight range) and measures
task-loss degradation, for N in {8, 4, 2} bits. Lower N = more aggressive
quantization.

Learned from the 2026-09-23/24 lesion-robustness episode (EXPERIMENTS.md):
a naive ratio-to-own-baseline metric is a known artifact when the real
and null networks start from very different baselines. This script
reports BOTH the ratio and the absolute post-quantization MSE from the
start, and the official gate is computed on the ABSOLUTE metric.

Same substrate/null/stats protocol otherwise. n=8 seeds, degree_preserving_rewire null.
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
BIT_LEVELS = [8, 4, 2]
GATE_BITS = 4
SEEDS = list(range(8))


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
    """Uniform signed fixed-point quantization of the effective recurrent
    weights (sign * softplus(gain)), applied only at eval time -- gain
    parameters are restored after. levels = 2^n_bits signed steps over
    each network's own [-max|w|, +max|w|] range (per-network scale, the
    fairest choice given real/null start from very different weight
    distributions)."""
    original_gain = brain.gain.data.clone()
    with torch.no_grad():
        w = brain.effective_weights()
        w_max = w.abs().max().clamp(min=1e-8)
        levels = 2 ** n_bits
        step = (2 * w_max) / levels
        w_q = torch.round(w / step) * step
        w_q = w_q.clamp(-w_max, w_max)
        # push the quantized weight back through sign*softplus(gain): gain = softplus^-1(|w_q|)
        mag_q = w_q.abs().clamp(min=1e-6)
        new_gain = mag_q + torch.log(-torch.expm1(-mag_q))
        brain.gain.data.copy_(new_gain)
        brain.sign.copy_(torch.sign(w_q) + (w_q == 0).float() * brain.sign)
        mse = eval_mse(brain, X_t, Y_t)
    brain.gain.data.copy_(original_gain)
    return mse


def gate_stats(real, null):
    real = np.array(real)
    null = np.array(null)
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(delta) > 0.33
    return {"real_median": float(np.median(real)), "null_median": float(np.median(null)),
            "wilcoxon_p": float(w_p), "mannwhitney_p": float(mw_p), "cliffs_delta": delta, "gate_pass": bool(gate)}


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

    log_path = "C:/projeler/fly_op/results/fly_quantization_robustness_multiseed.jsonl"
    per_bits = {b: {"real": [], "null": []} for b in BIT_LEVELS}
    baselines = {"real": [], "null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            X_test_t = torch.as_tensor(X_test, device=DEVICE)
            Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed}
            for name, sub in [("real", sub_real), ("null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                baseline = eval_mse(brain, X_test_t, Y_test_t)
                baselines[name].append(baseline)
                row[f"{name}_baseline"] = baseline
                for n_bits in BIT_LEVELS:
                    q_mse = quantized_mse(brain, X_test_t, Y_test_t, n_bits)
                    per_bits[n_bits][name].append(q_mse)
                    row[f"{name}_mse_{n_bits}bit"] = q_mse
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed}: {row}")

    summary = {"baseline": gate_stats(baselines["real"], baselines["null"])}
    print("\n=== SONUC (agirlik kuantizasyonu direnci, n=8, MUTLAK MSE) ===")
    print(f"referans (kuantizasyonsuz): {summary['baseline']}")
    for n_bits in BIT_LEVELS:
        s = gate_stats(per_bits[n_bits]["real"], per_bits[n_bits]["null"])
        summary[f"{n_bits}bit"] = s
        star = " <== RESMI KAPI" if n_bits == GATE_BITS else ""
        print(f"{n_bits}-bit{star}: real med={s['real_median']:.4f} null med={s['null_median']:.4f} "
              f"MW p={s['mannwhitney_p']:.5f} delta={s['cliffs_delta']:.3f} gate={'PASS' if s['gate_pass'] else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_quantization_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/fly_quantization_robustness_summary.json")


if __name__ == "__main__":
    main()
