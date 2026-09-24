"""Second physical/engineering benchmark, tested against the official
primary null (2026-09-21, user: "benchmarklari denedin mi? onlari
yapabilirsin" -- Gemini's structural-surrogate suggestion). Cantilever
beam cross-section design (benchmarks_structural.py): given a beam's
(width, height), predict the direction toward the minimum-cost
(stress + material) design -- same single-shot toplu-regresyon framing,
same RateBrain architecture, same encode/decode pipeline as
fly_slope_stability_degreenull_multiseed.py, but a genuinely different
physical domain (structural mechanics, not geotechnical slope stability).

Tests whether the modularity-related advantage (EXPERIMENTS.md
2026-09-20/21) generalizes across TASK types within the same species,
complementary to the C. elegans test (which asked about generalizing
across SPECIES). Runs against degree_preserving_rewire directly
(PROTOCOL.md's designated primary null) for n=8 first.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_structural import DIM, beam_cost, random_geometry, sample_valid_point
from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
    train,
)

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 0.02  # beam params live in [0.03, 0.6] m -- proportionally small probe step
SEEDS = list(range(8))


def build_training_set_geo(n_problems: int, n_starts: int, seed: int):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = random_geometry(rng)
        f = lambda p, geo=geo: beam_cost(p, geo)
        for _ in range(n_starts):
            x = sample_valid_point(geo, rng)
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
    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/beam_design_degreenull_multiseed.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            for name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)

                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test))
                    test_mse = float(((pred - torch.as_tensor(Y_test)) ** 2).mean().item())

                results[name].append(test_mse)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": test_mse}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:24s}] test_mse={test_mse:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["degree_preserving_null"])
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (konsol kiris tasarimi, degree_preserving null, n=8 tohum) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (mw_p < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/beam_design_degreenull_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/beam_design_degreenull_summary.json")


if __name__ == "__main__":
    main()
