"""Few-shot data-efficiency test (2026-09-23/24, overnight autonomous
battery). Same slope-stability task/architecture as the official result,
but trained on a drastically reduced training set (n_problems=2,
n_starts=3 = 6 total examples, vs the official 20x10=200) for the same
300 epochs, then evaluated on a large held-out set built the normal way.
Tests whether the real connectome's structure gives it a data-efficiency
edge (learns the task's "shape" faster from fewer examples) independent
of its full-data ceiling performance (already established, delta=0.884).
n=8 seeds, same null-comparison protocol.
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
SEEDS = list(range(8))
FEWSHOT_N_PROBLEMS = 2
FEWSHOT_N_STARTS = 3  # 6 total training examples, vs official 200


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

    log_path = "C:/projeler/fly_op/results/fly_fewshot_efficiency_multiseed.jsonl"
    results = {"real": [], "null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=FEWSHOT_N_PROBLEMS, n_starts=FEWSHOT_N_STARTS, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=20, n_starts=10, seed=19500 + seed)
            X_test_t = torch.as_tensor(X_test, device=DEVICE)
            Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed, "n_train_examples": len(X)}
            for name, sub in [("real", sub_real), ("null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    pred = brain(X_test_t)
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

    print(f"\n=== SONUC (az-ornekli veri-verimliligi, {FEWSHOT_N_PROBLEMS*FEWSHOT_N_STARTS} egitim ornegi, n=8) ===")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_fewshot_efficiency_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_fewshot_efficiency_summary.json")


if __name__ == "__main__":
    main()
