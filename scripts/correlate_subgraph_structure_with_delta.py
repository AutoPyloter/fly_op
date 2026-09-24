"""What makes a subgraph work? (2026-09-21, autonomous follow-up to the
subgraph-seed sweep, which found 9/11 independent alt-graf draws positive
but with highly variable magnitude -- 0.25 to 1.00). Tests the user's
"regional specialization" hypothesis directly: does a subgraph's
structural profile (density, modularity, clustering, mean degree)
predict its measured Cliff's delta?

For each of the 11 subgraph seeds tested this session (9000-9010),
rebuilds the exact same female FlyWire subgraph (same
select_connected_encode_decode/build_subgraph_bfs procedure) and computes
its structural stats -- no training needed, seconds not hours -- then
correlates each stat against the already-measured delta.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
from scipy import sparse, stats

from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"

# delta values already measured this session (see EXPERIMENTS.md 2026-09-21 entries)
KNOWN_DELTAS = {
    9000: 0.884,   # official, n=30
    9001: -0.219,  # n=8
    9002: 0.250,   # n=8
    9003: 0.250, 9004: -0.250, 9005: 1.000, 9006: 0.375,
    9007: 0.750, 9008: 0.625, 9009: 0.375, 9010: 1.000,  # n=4 each
}


def analyze_structure(seed, n_encode=6, n_decode=30, subgraph_size=3000):
    w = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    encode_full, decode_full = select_connected_encode_decode(
        w, afferent, efferent, n_encode=n_encode, n_decode=n_decode,
        max_hops=6, n_encode_candidates=200, seed=seed,
    )
    sub, encode_idx, decode_idx, nodes = build_subgraph_bfs(w, encode_full, decode_full, subgraph_size, seed=seed)

    n = sub.shape[0]
    coo = sub.tocoo()
    density = sub.nnz / (n * (n - 1))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = set((min(r, c), max(r, c)) for r, c in zip(coo.row.tolist(), coo.col.tolist()) if r != c)
    G.add_edges_from(edges)
    comm = nx.algorithms.community.louvain_communities(G, seed=0, resolution=1.0)
    Q = nx.algorithms.community.modularity(G, comm)
    clustering = nx.average_clustering(G)
    degrees = np.array([d for _, d in G.degree()])

    return {
        "seed": seed, "density": density, "modularity_Q": Q, "avg_clustering": clustering,
        "mean_degree": float(degrees.mean()), "n_communities": len(comm),
        "edges": int(sub.nnz),
    }


def main():
    rows = []
    for seed, delta in KNOWN_DELTAS.items():
        stats_row = analyze_structure(seed)
        stats_row["delta"] = delta
        rows.append(stats_row)
        print(f"seed={seed}: delta={delta:+.3f}  density={stats_row['density']:.5f}  "
              f"Q={stats_row['modularity_Q']:.4f}  clustering={stats_row['avg_clustering']:.4f}  "
              f"mean_degree={stats_row['mean_degree']:.1f}")

    with open("C:/projeler/fly_op/results/subgraph_structure_vs_delta.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    deltas = np.array([r["delta"] for r in rows])
    print("\n=== KORELASYON (delta ile yapisal ozellikler, n=11, Spearman) ===")
    for feat in ["density", "modularity_Q", "avg_clustering", "mean_degree", "n_communities", "edges"]:
        vals = np.array([r[feat] for r in rows])
        rho, p = stats.spearmanr(vals, deltas)
        print(f"  {feat:16s}: spearman_rho={rho:+.3f}  p={p:.3f}")

    print("\nwrote results/subgraph_structure_vs_delta.json")


if __name__ == "__main__":
    main()
