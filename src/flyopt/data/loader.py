"""Load FlyWire v783 (Zenodo 10.5281/zenodo.10676866, Dorkenwald et al.
2024) into a signed weighted sparse adjacency.

Frozen-snapshot rationale: the Codex live download portal serves a
continually-updated database, not a fixed version; the user chose the
Zenodo v783 snapshot instead, for reproducibility (EXPERIMENTS.md,
2026-09-14). See data/raw/README.md for the exact files/URLs.

Expected input:
  - proofread_connections_783.feather: one row per (pre, post, neuropil)
    triple, columns pre_pt_root_id, post_pt_root_id, neuropil, syn_count,
    and per-neurotransmitter average probability columns
    (gaba_avg, ach_avg, glut_avg, oct_avg, ser_avg, da_avg).
  - proofread_root_ids_783.npy: canonical array of all proofread neuron
    ids — this, not the connections file, defines n_units, since some
    proofread neurons have no recorded outgoing/incoming synapses.

E/I SIGN CONVENTION (flagged, not hidden): a neuron's excitatory/
inhibitory identity is derived per-neuron (Dale's law) as the
neurotransmitter with the largest syn_count-weighted total probability
across all of that neuron's outgoing rows, then mapped to a sign via
NT_COL_TO_SIGN. This is a domain modeling choice, not a measured fact —
ACH/DA/OCT/SER -> excitatory, GABA/GLUT -> inhibitory is a simplifying
default used elsewhere in fly connectome literature, but glutamate's sign
is context-dependent. Verify against literature/map.md before any Faz 1
result depends on it. A neuron with no outgoing edges gets sign 0.0
(undefined) rather than a guessed default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

NT_COLUMNS = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]

NT_COL_TO_SIGN = {
    "ach_avg": 1.0,
    "da_avg": 1.0,
    "oct_avg": 1.0,
    "ser_avg": 1.0,
    "gaba_avg": -1.0,
    "glut_avg": -1.0,
}


def load_connections(path) -> pd.DataFrame:
    return pd.read_feather(path)


def load_root_ids(path) -> np.ndarray:
    return np.load(path)


def _neuron_signs(connections: pd.DataFrame) -> pd.Series:
    """One sign per pre_pt_root_id, via syn_count-weighted dominant NT."""
    weighted = connections[NT_COLUMNS].multiply(connections["syn_count"], axis=0)
    weighted["pre_pt_root_id"] = connections["pre_pt_root_id"].to_numpy()
    nt_totals = weighted.groupby("pre_pt_root_id")[NT_COLUMNS].sum()
    dominant_nt = nt_totals.idxmax(axis=1)
    return dominant_nt.map(NT_COL_TO_SIGN)


def build_adjacency(
    connections: pd.DataFrame, root_ids: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    """Returns (weights_csr, source_sign, id_to_index).

    weights_csr[i, j] = signed synapse count from neuron j onto neuron i
    (pre-transposed for `W @ activity`, matching sim/lif_vectorized.py's
    convention). n = len(root_ids), i.e. every proofread neuron gets a row
    even if it has no synapses in `connections`.
    """
    id_to_index = {int(v): i for i, v in enumerate(root_ids)}
    n = len(id_to_index)

    signed_by_id = _neuron_signs(connections)

    edges = connections.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)[
        "syn_count"
    ].sum()

    unknown_pre = set(edges["pre_pt_root_id"]) - set(id_to_index)
    unknown_post = set(edges["post_pt_root_id"]) - set(id_to_index)
    if unknown_pre or unknown_post:
        raise ValueError(
            f"{len(unknown_pre)} pre and {len(unknown_post)} post ids in connections "
            "are absent from proofread_root_ids_783.npy — id lists are out of sync, "
            "re-download both files from the same Zenodo record."
        )

    rows = edges["post_pt_root_id"].map(id_to_index).to_numpy()  # i = post
    cols = edges["pre_pt_root_id"].map(id_to_index).to_numpy()  # j = pre
    sign = edges["pre_pt_root_id"].map(signed_by_id).fillna(0.0).to_numpy()
    data = edges["syn_count"].to_numpy(dtype=np.float64) * sign

    weights = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    weights.sum_duplicates()

    source_sign = np.zeros(n)
    for root_id, s in signed_by_id.items():
        idx = id_to_index.get(int(root_id))
        if idx is not None:
            source_sign[idx] = s

    return weights, source_sign, id_to_index
