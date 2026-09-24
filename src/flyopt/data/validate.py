"""CLI: python -m flyopt.data.validate

Measures network statistics from the downloaded FlyWire v783 Zenodo
snapshot and writes data/processed/manifest.json. Never copies published
numbers — every value here is computed from the files actually present in
data/raw/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from flyopt.data.loader import build_adjacency, load_connections, load_root_ids

DATA_RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"

REQUIRED_FILES = [
    "proofread_connections_783.feather",
    "proofread_root_ids_783.npy",
]


def main() -> int:
    missing = [f for f in REQUIRED_FILES if not (DATA_RAW / f).exists()]
    if missing:
        print(
            "Missing FlyWire data files in data/raw/: "
            + ", ".join(missing)
            + "\nSee data/raw/README.md for download instructions "
            "(Zenodo record 10.5281/zenodo.10676866, license/citation "
            "acceptance required — user action).",
            file=sys.stderr,
        )
        return 1

    connections = load_connections(DATA_RAW / "proofread_connections_783.feather")
    root_ids = load_root_ids(DATA_RAW / "proofread_root_ids_783.npy")
    weights, source_sign, id_to_index = build_adjacency(connections, root_ids)

    n = weights.shape[0]
    density = weights.nnz / (n * n)
    out_deg = np.asarray((weights != 0).sum(axis=0)).flatten()
    in_deg = np.asarray((weights != 0).sum(axis=1)).flatten()
    n_no_sign = int((source_sign == 0.0).sum())

    manifest = {
        "n_neurons": n,
        "n_edges": int(weights.nnz),
        "density": density,
        "frac_excitatory": float((source_sign > 0).mean()),
        "frac_inhibitory": float((source_sign < 0).mean()),
        "n_neurons_no_outgoing_sign": n_no_sign,
        "out_degree_mean": float(out_deg.mean()),
        "out_degree_max": int(out_deg.max()),
        "in_degree_mean": float(in_deg.mean()),
        "in_degree_max": int(in_deg.max()),
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sparse_npz_path = DATA_PROCESSED / "adjacency.npz"
    from scipy import sparse as sp

    sp.save_npz(sparse_npz_path, weights)
    np.save(DATA_PROCESSED / "source_sign.npy", source_sign)

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {sparse_npz_path} and data/processed/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
