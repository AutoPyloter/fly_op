"""Modularity dose-response test (2026-09-21): does the RECOVERED
advantage (Cliff's delta) scale with HOW MUCH modularity is preserved?
community_preserving_rewire at resolution=1.0 (Q=0.357, matching real's
0.357) gave delta=0.429 (n=30). Here, same null-generation mechanism but
at two other Louvain resolutions -- 0.5 (coarse, only 5 communities,
Q=0.088, close to degree_preserving's near-zero) and 2.0 (finer, 20
communities, Q=0.315, still substantial) -- pilot n=8 each. If delta
increases monotonically with preserved Q, that's independent confirming
evidence for the modularity mechanism (EXPERIMENTS.md 2026-09-20/21).
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import community_preserving_rewire
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
RAY_RADIUS = 1.0
SEEDS = list(range(8))
RESOLUTIONS = [0.5, 2.0]  # 1.0 already done (delta=0.429, n=30)


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


def compute_communities(sub: sparse.csr_matrix, resolution: float) -> tuple[np.ndarray, float]:
    coo = sub.tocoo()
    n = sub.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = set((min(r, c), max(r, c)) for r, c in zip(coo.row.tolist(), coo.col.tolist()) if r != c)
    G.add_edges_from(edges)
    comm_sets = nx.algorithms.community.louvain_communities(G, seed=0, resolution=resolution)
    modularity = nx.algorithms.community.modularity(G, comm_sets)
    communities = np.zeros(n, dtype=np.int64)
    for cid, members in enumerate(comm_sets):
        for m in members:
            communities[m] = cid
    print(f"resolution={resolution}: {len(comm_sets)} communities, modularity={modularity:.4f}")
    return communities, modularity


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

    all_results = {}
    for resolution in RESOLUTIONS:
        communities, modularity = compute_communities(sub_real, resolution)
        results = {"real_connectome": [], "community_preserving_null": []}
        log_path = f"C:/projeler/fly_op/results/fly_slope_stability_modularity_res{resolution}_multiseed.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for seed in SEEDS:
                X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
                X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
                sub_null = community_preserving_rewire(sub_real, communities, seed=seed)

                for name, sub in [("real_connectome", sub_real), ("community_preserving_null", sub_null)]:
                    brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed)
                    train(brain, X, Y, epochs=EPOCHS, lr=LR)
                    with torch.no_grad():
                        pred = brain(torch.as_tensor(X_test))
                        test_mse = float(((pred - torch.as_tensor(Y_test)) ** 2).mean().item())
                    results[name].append(test_mse)
                    f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": test_mse}) + "\n")
                    f.flush()
                    print(f"res={resolution} seed={seed} [{name:26s}] test_mse={test_mse:.4f}")

        real = np.array(results["real_connectome"])
        null = np.array(results["community_preserving_null"])
        w_stat, w_p = stats.wilcoxon(real, null)
        mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
        gt = np.sum(real[:, None] < null[None, :])
        lt = np.sum(real[:, None] > null[None, :])
        cliffs_delta = float((gt - lt) / (len(real) * len(null)))
        print(f"\n=== resolution={resolution} (Q={modularity:.4f}), n=8 ===")
        print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
        print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}\n")
        all_results[f"resolution_{resolution}"] = {
            "modularity": modularity, "real": real.tolist(), "null": null.tolist(),
            "wilcoxon_p": float(w_p), "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta,
        }

    with open("C:/projeler/fly_op/results/fly_slope_stability_modularity_doseresponse_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\n=== DOZ-YANIT OZETI ===")
    print(f"resolution=0.5 (dusuk Q): delta={all_results['resolution_0.5']['cliffs_delta']:.3f}")
    print(f"resolution=1.0 (Q=0.357, onceki sonuc): delta=0.429 (n=30)")
    print(f"resolution=2.0 (Q~0.31): delta={all_results['resolution_2.0']['cliffs_delta']:.3f}")
    print("wrote results/fly_slope_stability_modularity_doseresponse_summary.json")


if __name__ == "__main__":
    main()
