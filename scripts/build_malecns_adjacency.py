"""Builds a fly_op-compatible adjacency + afferent/efferent pools from the
male Drosophila CNS connectome (2026-09-21, user request: "aynı çalışmayı
bir de erkek sinek beyni üzerinden ilerletelim" -- run our own official
methodology, not the separate fly_demos/malecns project's own DAgger/F1
pipeline, on the MALE connectome as a second cross-individual/sex
generalization check, alongside the C. elegans cross-species one).

Source data already downloaded (no new download needed) in the sibling
fly_demos/malecns project: data/graph_w5/{edges.npz,neurons.parquet}.
edges.npz's weight is ALREADY signed (unlike our own FlyWire loader, which
had to derive sign from neurotransmitter columns itself).

Afferent/efferent pools are read from neurons.parquet's `superclass`
column (verified index-aligned with edges.npz's pre/post integers), using
the PRIMARY sensory/motor categories only (not relay categories like
visual_projection/ascending/descending) -- matching FlyWire's flow-based
afferent/efferent definition in spirit (see
data/raw/Supplemental_file1_neuron_annotations.tsv's `flow` column, used
by scripts/add_edges_to_activation_3d.py's verification).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

MALECNS_SRC = "C:/projeler/fly_demos/malecns/data/graph_w5"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"

AFFERENT_SUPERCLASSES = {
    "ol_sensory", "cb_sensory", "vnc_sensory",
    "sensory_ascending", "sensory_descending",
    "cb_sensory_tbc", "vnc_sensory_tbc", "sensory_ascending_tbc",
}
EFFERENT_SUPERCLASSES = {
    "vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent",
    "efferent_ascending", "efferent_descending",
}


def main():
    edges = np.load(f"{MALECNS_SRC}/edges.npz")
    neurons = pd.read_parquet(f"{MALECNS_SRC}/neurons.parquet")
    n = len(neurons)
    assert int(max(edges["pre"].max(), edges["post"].max())) + 1 <= n, "index/row mismatch"

    weights = sparse.csr_matrix(
        (edges["weight"].astype(np.float64), (edges["post"], edges["pre"])), shape=(n, n)
    )
    weights.sum_duplicates()
    print(f"male CNS adjacency: {n} nodes, {weights.nnz} directed edges")

    sc = neurons["superclass"]
    afferent_idx = np.where(sc.isin(AFFERENT_SUPERCLASSES))[0].astype(np.int64)
    efferent_idx = np.where(sc.isin(EFFERENT_SUPERCLASSES))[0].astype(np.int64)
    print(f"afferent pool: {len(afferent_idx)}   efferent pool: {len(efferent_idx)}")

    sparse.save_npz(f"{DATA_PROCESSED}/malecns_adjacency.npz", weights)
    np.save(f"{DATA_PROCESSED}/malecns_afferent_indices.npy", afferent_idx)
    np.save(f"{DATA_PROCESSED}/malecns_efferent_indices.npy", efferent_idx)
    print(f"wrote malecns_adjacency.npz, malecns_afferent_indices.npy, malecns_efferent_indices.npy to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
