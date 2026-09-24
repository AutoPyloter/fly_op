"""Input-scale (sensing radius) generalization test (2026-09-24). Trains
real vs null RateBrain on the official slope-stability task at the
official RAY_RADIUS=1.0 (identical recipe to the official n=30 result),
then evaluates the SAME trained weights -- no further training -- on a
held-out set whose rays are recomputed at a DIFFERENT radius (0.1x and
5x the training radius). Tests whether the connectome's advantage is
tied to the specific sensing scale it was trained at, or generalizes to
a finer/coarser sensing radius zero-shot. Unlike cross-task-transfer
(different objective function), this keeps the SAME task/objective and
only changes the finite-difference probe radius at evaluation.

Same substrate/null/stats protocol as every official test. n=8 seeds,
degree_preserving_rewire null.
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
TRAIN_RAY_RADIUS = 1.0
EVAL_RADII = [0.1, 1.0, 5.0]  # 0.1x, 1x (reference), 5x the training radius
SEEDS = list(range(8))


def build_set(n_problems: int, n_starts: int, seed: int, ray_radius: float):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = random_geometry(rng)
        lo, hi = geo.param_bounds()
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi)
            fx = f(x)
            rays = _rays(x, f, fx, ray_radius)
            target = _teacher_delta(x, rays, ray_radius)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(DIM)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


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

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=TRAIN_RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_rayradius_generalization_multiseed.jsonl"
    per_radius = {r: {"real": [], "null": []} for r in EVAL_RADII}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_set(n_problems=20, n_starts=10, seed=9500 + seed, ray_radius=TRAIN_RAY_RADIUS)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed}
            for name, sub in [("real", sub_real), ("null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                for radius in EVAL_RADII:
                    X_test, Y_test = build_set(n_problems=10, n_starts=10, seed=19500 + seed, ray_radius=radius)
                    X_test_t = torch.as_tensor(X_test, device=DEVICE)
                    Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
                    with torch.no_grad():
                        pred = brain(X_test_t)
                        mse = float(((pred - Y_test_t) ** 2).mean().item())
                    per_radius[radius][name].append(mse)
                    row[f"{name}_mse_r{radius}"] = mse
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed}: {row}")

    summary = {}
    print("\n=== SONUC (girdi-olcegi genellemesi, n=8) ===")
    for radius in EVAL_RADII:
        real = np.array(per_radius[radius]["real"])
        null = np.array(per_radius[radius]["null"])
        w_stat, w_p = stats.wilcoxon(real, null)
        mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
        gt = np.sum(real[:, None] < null[None, :])
        lt = np.sum(real[:, None] > null[None, :])
        cliffs_delta = float((gt - lt) / (len(real) * len(null)))
        gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33
        summary[str(radius)] = {"real_median": float(np.median(real)), "null_median": float(np.median(null)),
                                 "wilcoxon_p": float(w_p), "mannwhitney_p": float(mw_p),
                                 "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}
        tag = " <== egitim yaricapi (referans)" if radius == TRAIN_RAY_RADIUS else ""
        print(f"r={radius}{tag}: real med={np.median(real):.4f} null med={np.median(null):.4f} "
              f"MW p={mw_p:.5f} delta={cliffs_delta:.3f} gate={'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_rayradius_generalization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/fly_rayradius_generalization_summary.json")


if __name__ == "__main__":
    main()
