"""Augments activation_3d_demo.json with the subgraph's edge list (2026-09-21,
user feedback: 3D viz looks like a formless "cloud", wants it to look more
"linear" / wired -- i.e. show the actual connectome edges, not just points).

Rebuilds the exact same 3000-node subgraph as record_activation_3d.py (same
seeds, same selection code -> same node ordering, so edge indices line up
with the existing positions/frames already in the JSON) WITHOUT retraining,
and appends a compact edge list.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 6  # benchmarks_geo.DIM=3 -> 2*DIM rays, matches record_activation_3d.py
N_READOUT = 30
SUBGRAPH_SIZE = 3000


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"rebuilt subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} directed edges")

    coo = sub_real.tocoo()
    # undirected unique pairs, keeping the max |weight| seen for each pair
    # (drop self-loops; rendering all ~150k directed edges makes the scene
    # an unreadable haze -- keep only the strongest TOP_K as a legible
    # structural "skeleton", per user feedback that the viz looked like a
    # formless cloud and needed to look more wired/linear)
    best = {}
    for r, c, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if r == c:
            continue
        key = (r, c) if r < c else (c, r)
        aw = abs(w)
        if key not in best or aw > best[key]:
            best[key] = aw
    print(f"{len(best)} unique undirected edges (pre-filter)")

    TOP_K = 18000
    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:TOP_K]
    edges = sorted(k for k, _ in ranked)
    print(f"kept top {len(edges)} edges by |weight|")

    path = "C:/projeler/fly_op/results/activation_3d_demo.json"
    with open(path, "r", encoding="utf-8") as f:
        out = json.load(f)

    assert out["n_nodes"] == sub_real.shape[0], "node count mismatch -- selection did not reproduce identically"

    out["edges"] = edges
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {len(edges)} edges into {path}")


if __name__ == "__main__":
    main()
