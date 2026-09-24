"""Build a FlyOpt-compatible adjacency matrix + afferent/efferent index
arrays from the Cook et al. 2019 (Nature) C. elegans hermaphrodite
chemical connectome (downloaded from networks.skewed.de/net/celegans_2019,
2026-09-21 -- cross-species generalization test for the modularity
mechanism found on FlyWire, see EXPERIMENTS.md 2026-09-20/21).

Restricts to the classic "somatic nervous system" convention (sensory
neurons + interneurons + motor neurons, excluding pharynx/muscle/
sex-specific end organs) -- 272 of the 453 nodes -- matching FlyWire's
afferent(sensory)->efferent(motor) pipeline exactly: SENSORY NEURONS
become the afferent pool, MOTOR NEURONS the efferent pool.

Output convention matches flyopt's existing FlyWire processing:
adjacency[i, j] = signed weight of the edge j->i (row=post/target,
col=pre/source), scipy.sparse.csr_matrix.
"""
from __future__ import annotations

import csv

import numpy as np
from scipy import sparse

RAW_DIR = "C:/projeler/fly_op/data/raw/celegans"
OUT_DIR = "C:/projeler/fly_op/data/processed"

KEEP_TYPES = {"SENSORY NEURONS", "INTERNEURONS", "MOTOR NEURONS"}


def main() -> None:
    with open(f"{RAW_DIR}/nodes.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    node_type = {}
    node_name = {}
    for r in rows:
        idx, ntype, _subtype, name = int(r[0]), r[1], r[2], r[3]
        node_type[idx] = ntype
        node_name[idx] = name

    kept_orig_ids = sorted(i for i, t in node_type.items() if t in KEEP_TYPES)
    remap = {orig: new for new, orig in enumerate(kept_orig_ids)}
    n = len(kept_orig_ids)
    print(f"kept {n} / {len(node_type)} nodes: "
          f"{sum(1 for i in kept_orig_ids if node_type[i]=='SENSORY NEURONS')} sensory, "
          f"{sum(1 for i in kept_orig_ids if node_type[i]=='INTERNEURONS')} inter, "
          f"{sum(1 for i in kept_orig_ids if node_type[i]=='MOTOR NEURONS')} motor")

    with open(f"{RAW_DIR}/edges.csv", newline="", encoding="utf-8") as f:
        edge_rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]

    src, dst, weight = [], [], []
    dropped = 0
    for r in edge_rows:
        s, t, w = int(r[0]), int(r[1]), float(r[2])
        if s in remap and t in remap:
            src.append(remap[s])
            dst.append(remap[t])
            weight.append(w)  # chemical synapses are excitatory-only in this dataset; sign left positive
        else:
            dropped += 1
    print(f"kept {len(src)} edges, dropped {dropped} (endpoint outside somatic nervous system)")

    # adjacency[i, j] = weight of edge j -> i  =>  row=dst(post), col=src(pre)
    adjacency = sparse.csr_matrix((weight, (dst, src)), shape=(n, n))
    sparse.save_npz(f"{OUT_DIR}/celegans_adjacency.npz", adjacency)

    sensory_idx = np.array([remap[i] for i in kept_orig_ids if node_type[i] == "SENSORY NEURONS"], dtype=np.int64)
    motor_idx = np.array([remap[i] for i in kept_orig_ids if node_type[i] == "MOTOR NEURONS"], dtype=np.int64)
    inter_idx = np.array([remap[i] for i in kept_orig_ids if node_type[i] == "INTERNEURONS"], dtype=np.int64)
    np.save(f"{OUT_DIR}/celegans_sensory_indices.npy", sensory_idx)
    np.save(f"{OUT_DIR}/celegans_motor_indices.npy", motor_idx)
    np.save(f"{OUT_DIR}/celegans_interneuron_indices.npy", inter_idx)

    out_deg = np.asarray((adjacency != 0).sum(axis=0)).ravel()
    in_deg = np.asarray((adjacency != 0).sum(axis=1)).ravel()
    print(f"n={n} edges={adjacency.nnz} mean_out_deg={out_deg.mean():.2f} mean_in_deg={in_deg.mean():.2f}")
    print(f"out_std/mean={out_deg.std()/out_deg.mean():.3f} in_std/mean={in_deg.std()/in_deg.mean():.3f}")
    print(f"wrote {OUT_DIR}/celegans_adjacency.npz (+ sensory/motor/interneuron index arrays)")


if __name__ == "__main__":
    main()
