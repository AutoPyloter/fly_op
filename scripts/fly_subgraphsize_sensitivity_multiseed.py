"""Subgraph-size sensitivity test (2026-09-24). The official slope-
stability finding (delta=0.884, n=30) always used a fixed SUBGRAPH_SIZE
of 3000 nodes. This tests whether the advantage is an artifact of that
specific size, or holds (and how its magnitude scales) at smaller
(1000) and larger (6000, 10000) subgraphs -- same encode/decode seed
(9000), same training recipe, only SUBGRAPH_SIZE varies. n=8 pilot per
size, degree_preserving_rewire null, same stats convention.
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
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SUBGRAPH_SIZES = [1000, 3000, 6000, 10000]  # 3000 = official reference size
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


def run_for_size(subgraph_size: int, base_weights, afferent, efferent):
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, subgraph_size, seed=9000)
    print(f"  size={subgraph_size}: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = f"C:/projeler/fly_op/results/fly_subgraphsize_{subgraph_size}_multiseed.jsonl"
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
                with torch.no_grad():
                    pred = brain(X_test_t)
                    mse = float(((pred - Y_test_t) ** 2).mean().item())
                store.append(mse)
                row[f"{name}_test_mse"] = mse
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
    return gate_stats(real_mses, null_mses)


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    summary = {}
    print("=== SONUC (alt-graf boyutu duyarliligi, n=8 her boyut) ===")
    for size in SUBGRAPH_SIZES:
        s = run_for_size(size, base_weights, afferent, efferent)
        summary[str(size)] = s
        tag = " <== resmi referans boyut" if size == 3000 else ""
        print(f"size={size}{tag}: real med={s['real_median']:.4f} null med={s['null_median']:.4f} "
              f"MW p={s['mannwhitney_p']:.5f} delta={s['cliffs_delta']:.3f} gate={'PASS' if s['gate_pass'] else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_subgraphsize_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/fly_subgraphsize_sensitivity_summary.json")


if __name__ == "__main__":
    main()
