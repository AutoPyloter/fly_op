"""Structural diagnostic for null models (2026-09-19, following the
discovery that er_null's "advantage" for both malecns and CX-modulated
was very likely a degree-heterogeneity artifact, not genuine wiring
specificity -- see EXPERIMENTS.md "MEKANIZMA BULUNDU"). Compares
out/in-degree heterogeneity (std/mean ratio) and reciprocity (fraction
of edges with a reverse edge present) across real_connectome, er_null,
degree_preserving_rewire, and weight_shuffle on any given subgraph.

Run this on any new subgraph BEFORE trusting an er_null-only comparison:
if real's degree std/mean ratio is much higher than er_null's (as it
almost always will be for a FlyWire subgraph), an er_null-only "positive"
finding should be treated as provisional until also checked against
degree_preserving_rewire.

Usage: .venv/Scripts/python.exe scripts/analyze_null_model_structure.py
(edit SUBGRAPH_ARGS below to point at a different encode/decode pool
selection, or call `stats()` directly on any scipy.sparse matrix).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from flyopt.substrates.graph_builders import degree_preserving_rewire, er_null, weight_shuffle
from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
CELLTYPES = f"{DATA_PROCESSED}/celltypes"


def stats(sub: sparse.spmatrix, name: str) -> None:
    coo = sub.tocoo()
    n = sub.shape[0]
    out_deg = np.bincount(coo.row, minlength=n)
    in_deg = np.bincount(coo.col, minlength=n)
    edge_set = set(zip(coo.row.tolist(), coo.col.tolist()))
    recip = sum(1 for r, c in zip(coo.row, coo.col) if (c, r) in edge_set)
    print(f"{name:20s} n={n} edges={coo.nnz} "
          f"out_std/mean={out_deg.std() / out_deg.mean():.3f} "
          f"in_std/mean={in_deg.std() / in_deg.mean():.3f} "
          f"max_out={out_deg.max()} max_in={in_deg.max()} "
          f"reciprocity={recip / coo.nnz:.4f}")


def compare_all_nulls(sub_real: sparse.csr_matrix, seed: int = 0) -> None:
    stats(sub_real, "real")
    stats(er_null(sub_real, seed=seed), "er_null")
    stats(degree_preserving_rewire(sub_real, seed=seed), "degree_preserving")
    stats(weight_shuffle(sub_real, seed=seed), "weight_shuffle")


def main() -> None:
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    cx_pool = np.load(f"{CELLTYPES}/central_complex_indices.npy")

    encode_full, cx_verified = select_connected_encode_decode(
        base_weights, afferent, cx_pool, n_encode=4, n_decode=35,
        max_hops=6, n_encode_candidates=200, seed=16000,
    )
    decode_full = cx_verified[:20]
    cx_full = cx_verified[20:35]
    all_targets = np.concatenate([decode_full, cx_full])
    sub_real, _encode_idx, _mapped_targets, _nodes = build_subgraph_bfs(
        base_weights, encode_full, all_targets, 2000, seed=16000
    )
    compare_all_nulls(sub_real, seed=0)


if __name__ == "__main__":
    main()
