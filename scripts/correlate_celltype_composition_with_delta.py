"""Cell-type/region composition vs delta (2026-09-22, autonomous follow-up:
coarse graph-theoretic structural metrics -- density/modularity/clustering
-- showed no significant correlation with delta across the 11 subgraph
seeds tested (EXPERIMENTS.md 2026-09-21). This tests a more specific
hypothesis: does the CELL-TYPE COMPOSITION of a subgraph (fraction of
nodes belonging to each FlyWire super_class -- optic, central, sensory,
visual_projection, ascending, descending, motor, etc.) predict delta?
Directly tests the user's "regional specialization" idea at a finer
resolution than raw graph statistics.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse, stats

from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DATA_RAW = "C:/projeler/fly_op/data/raw"

KNOWN_DELTAS = {
    9000: 0.884, 9001: -0.219, 9002: 0.250, 9003: 0.250, 9004: -0.250,
    9005: 1.000, 9006: 0.375, 9007: 0.750, 9008: 0.625, 9009: 0.375, 9010: 1.000,
}


def main():
    w = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    root_ids = np.load(f"{DATA_RAW}/proofread_root_ids_783.npy")

    ann = pd.read_csv(f"{DATA_RAW}/Supplemental_file1_neuron_annotations.tsv", sep="\t",
                       usecols=["root_id", "super_class"])
    ann = ann.drop_duplicates(subset="root_id").set_index("root_id")

    rows = []
    for seed, delta in KNOWN_DELTAS.items():
        encode_full, decode_full = select_connected_encode_decode(
            w, afferent, efferent, n_encode=6, n_decode=30, max_hops=6, n_encode_candidates=200, seed=seed,
        )
        sub, encode_idx, decode_idx, nodes = build_subgraph_bfs(w, encode_full, decode_full, 3000, seed=seed)
        node_root_ids = root_ids[nodes]
        sc = ann.reindex(node_root_ids)["super_class"]
        frac = (sc.value_counts(normalize=True, dropna=False)).to_dict()
        row = {"seed": seed, "delta": delta, **{f"frac_{k}": v for k, v in frac.items()}}
        rows.append(row)
        top3 = sorted(frac.items(), key=lambda kv: -kv[1])[:3]
        print(f"seed={seed}: delta={delta:+.3f}  top3={top3}")

    df = pd.DataFrame(rows).fillna(0.0)
    df.to_csv("C:/projeler/fly_op/results/subgraph_celltype_vs_delta.csv", index=False)

    print("\n=== KORELASYON (delta ile super_class oranlari, n=11, Spearman) ===")
    frac_cols = [c for c in df.columns if c.startswith("frac_")]
    results = {}
    for col in frac_cols:
        if df[col].std() < 1e-9:
            continue
        rho, p = stats.spearmanr(df[col], df["delta"])
        results[col] = {"rho": float(rho), "p": float(p)}
        print(f"  {col:28s}: rho={rho:+.3f}  p={p:.3f}")

    with open("C:/projeler/fly_op/results/subgraph_celltype_vs_delta_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results/subgraph_celltype_vs_delta.csv and _summary.json")


if __name__ == "__main__":
    main()
