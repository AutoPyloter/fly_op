"""Follow-up to fly_noise_robustness_official30.py: that script's n=30
extension (seeds 8-29) only logged the RATIO (noisy/baseline), not the
absolute noisy/baseline MSE values, so the official n=30 gate (delta=-0.978)
could not be independently re-verified on absolute terms the way the n=8
pilot was (see EXPERIMENTS.md 2026-09-24 correction entry: lesion-robustness
turned out to be a ratio-metric artifact, noise-robustness held up under an
n=8 absolute-MSE spot-check but n=30 absolute confirmation was still
pending). This reruns seeds 8-29 with IDENTICAL training/eval protocol,
additionally logging real_baseline/real_noisy/null_baseline/null_noisy, and
combines with the n=8 pilot's already-available absolute values for the
official n=30 absolute-metric verdict.
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
SEEDS = list(range(8, 30))
NOISE_REL_STD = 0.15
N_NOISE_DRAWS = 5


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


def noisy_mse(brain, X_t, Y_t, rng_seed: int, n_draws: int) -> float:
    gen = torch.Generator(device=X_t.device).manual_seed(rng_seed)
    per_dim_std = X_t.std(dim=0, keepdim=True)
    mses = []
    for _ in range(n_draws):
        noise = torch.randn(X_t.shape, generator=gen, device=X_t.device) * (NOISE_REL_STD * per_dim_std)
        X_noisy = X_t + noise
        with torch.no_grad():
            pred = brain(X_noisy)
            mses.append(float(((pred - Y_t) ** 2).mean().item()))
    return float(np.mean(mses))


def gate_stats(real, null):
    real = np.array(real)
    null = np.array(null)
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(delta) > 0.33
    return {
        "real_median": float(np.median(real)), "null_median": float(np.median(null)),
        "wilcoxon_p": float(w_p), "mannwhitney_p": float(mw_p),
        "cliffs_delta": delta, "gate_pass": bool(gate),
    }


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

    log_path = "C:/projeler/fly_op/results/fly_noise_robustness_absolute_seeds8to29.jsonl"
    real_baseline, real_noisy, null_baseline, null_noisy = [], [], [], []
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
                noisy = noisy_mse(brain, X_test_t, Y_test_t, 80000 + seed, N_NOISE_DRAWS)
                row[f"{name}_baseline"] = baseline
                row[f"{name}_noisy"] = noisy
                if name == "real":
                    real_baseline.append(baseline); real_noisy.append(noisy)
                else:
                    null_baseline.append(baseline); null_noisy.append(noisy)
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed} done")

    orig = [json.loads(l) for l in open("C:/projeler/fly_op/results/fly_noise_robustness_multiseed.jsonl", encoding="utf-8")]
    real_noisy_all = [r["real_noisy"] for r in orig] + real_noisy
    null_noisy_all = [r["null_noisy"] for r in orig] + null_noisy
    assert len(real_noisy_all) == 30 and len(null_noisy_all) == 30

    stats_abs = gate_stats(real_noisy_all, null_noisy_all)
    print("\n=== SONUC (gurultu direnci, MUTLAK gurultulu MSE, n=30) ===")
    print(stats_abs)

    with open("C:/projeler/fly_op/results/fly_noise_robustness_absolute_official30_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real_noisy": real_noisy_all, "null_noisy": null_noisy_all, **stats_abs}, f, indent=2)
    print("wrote results/fly_noise_robustness_absolute_official30_summary.json")


if __name__ == "__main__":
    main()
