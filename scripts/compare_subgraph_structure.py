"""Structural diagnostic (2026-09-21, user asked: "neden dişi sineğin
yaptığını erkek sinek yapamadı, araştırdın mı, yanlış bir işlem mi
yaptık doğruladın mı?"). Rebuilds both the official female FlyWire FAFB
subgraph and the male CNS subgraph (same seed=9000, same
select_connected_encode_decode/build_subgraph_bfs procedure) and compares
their structural properties directly -- checks for a pipeline bug
(disconnected/degenerate subgraph, unverified encode/decode) AND reports
real structural differences (density, modularity, clustering) that might
explain the male CNS's negative slope-stability result
(EXPERIMENTS.md 2026-09-21).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import sparse

from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"


def analyze(name, adj_path, afferent_path, efferent_path, n_encode=6, n_decode=30, subgraph_size=3000, seed=9000):
    w = sparse.load_npz(f"{DATA_PROCESSED}/{adj_path}")
    afferent = np.load(f"{DATA_PROCESSED}/{afferent_path}")
    efferent = np.load(f"{DATA_PROCESSED}/{efferent_path}")
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
    n_isolated = int((degrees == 0).sum())

    print(f"\n=== {name} ===")
    print(f"full graph: {w.shape[0]} nodes, {w.nnz} edges | afferent pool={len(afferent)} efferent pool={len(efferent)}")
    print(f"encode/decode verified: {len(encode_idx)}/{n_encode} encode, {len(decode_idx)}/{n_decode} decode")
    print(f"subgraph: {n} nodes, {sub.nnz} directed edges, density={density:.5f}")
    print(f"modularity Q={Q:.4f}  avg_clustering={clustering:.4f}  n_communities={len(comm)}")
    print(f"degree: mean={degrees.mean():.2f} median={np.median(degrees):.1f} max={degrees.max()} isolated_nodes={n_isolated}")
    return dict(n=n, edges=int(sub.nnz), density=density, Q=Q, clustering=clustering,
                n_encode=len(encode_idx), n_decode=len(decode_idx), isolated=n_isolated)


def main():
    analyze("Disi FlyWire FAFB (resmi bulgu, seed=9000)", "adjacency.npz", "afferent_indices.npy", "efferent_indices.npy")
    analyze("Erkek CNS (seed=9000)", "malecns_adjacency.npz", "malecns_afferent_indices.npy", "malecns_efferent_indices.npy")


if __name__ == "__main__":
    main()
