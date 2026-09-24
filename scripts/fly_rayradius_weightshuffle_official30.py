"""n=8 pilot (fly_rayradius_nullfamily_mechanism.py, weight_shuffle
condition) showed delta=0.344, barely crossing the Cliff's-delta
magnitude threshold (|delta|>0.33) but far from p<0.05 (p=0.279). Per
standing protocol (magnitude alone triggers an n=30 extension), this
runs 22 new seeds (8-29) for weight_shuffle only and combines with the
original 0-7 for the official n=30 verdict.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import weight_shuffle
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
EVAL_RADIUS = 0.1
SEEDS = list(range(8, 30))


def build_set(n_problems, n_starts, seed, ray_radius):
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

    log_path = "C:/projeler/fly_op/results/fly_rayradius_weightshuffle_seeds8to29.jsonl"
    real_mses, null_mses = [], []
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_set(n_problems=20, n_starts=10, seed=9500 + seed, ray_radius=TRAIN_RAY_RADIUS)
            sub_null = weight_shuffle(sub_real, seed=seed)

            row = {"seed": seed}
            for name, sub, store in [("real", sub_real, real_mses), ("null", sub_null, null_mses)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                X_test, Y_test = build_set(n_problems=10, n_starts=10, seed=19500 + seed, ray_radius=EVAL_RADIUS)
                X_test_t = torch.as_tensor(X_test, device=DEVICE)
                Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
                with torch.no_grad():
                    pred = brain(X_test_t)
                    mse = float(((pred - Y_test_t) ** 2).mean().item())
                store.append(mse)
                row[f"{name}_mse"] = mse
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed}: {row}")

    orig = [json.loads(l) for l in open("C:/projeler/fly_op/results/fly_rayradius_nullfamily_mechanism.jsonl", encoding="utf-8")]
    orig_wsh = [r for r in orig if r["null"] == "weight_shuffle"]
    if not orig_wsh:
        # weight_shuffle pilot data lives in the part2 run
        orig_wsh = [json.loads(l) for l in open("C:/projeler/fly_op/results/fly_rayradius_nullfamily_mechanism_part2.jsonl", encoding="utf-8") if json.loads(l)["null"] == "weight_shuffle"]
    real = np.array([r["real_mse"] for r in orig_wsh] + real_mses)
    null = np.array([r["null_mse"] for r in orig_wsh] + null_mses)
    assert len(real) == 30 and len(null) == 30

    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33

    print("\n=== SONUC (girdi-olcegi, weight_shuffle null, n=30) ===")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Mann-Whitney p={mw_p:.6f}  Wilcoxon p={w_p:.6f}  Cliff's delta={cliffs_delta:.3f}  gate={'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_rayradius_weightshuffle_official30_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_rayradius_weightshuffle_official30_summary.json")


if __name__ == "__main__":
    main()
