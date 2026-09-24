"""Adds a full-brain neuron position backdrop to activation_3d_demo.json
(2026-09-21, user request: the demo-style FlyWire brain outline would make
the shape more recognizable than our 3000-node subgraph alone).

Uses ALL positions already present in the locally-downloaded
Supplemental_file1_neuron_annotations.tsv (139248 proofread neurons) --
no new FlyWire download needed. These are rendered client-side as a faint,
static silhouette behind the animated subgraph, giving real anatomical
context instead of a synthetic mesh.
"""
from __future__ import annotations

import json

import pandas as pd

DATA_RAW = "C:/projeler/fly_op/data/raw"


def main():
    ann = pd.read_csv(f"{DATA_RAW}/Supplemental_file1_neuron_annotations.tsv", sep="\t",
                       usecols=["root_id", "pos_x", "pos_y", "pos_z"])
    ann = ann.drop_duplicates(subset="root_id")
    backdrop = ann[["pos_x", "pos_y", "pos_z"]].astype(int).values.tolist()
    print(f"{len(backdrop)} full-brain backdrop positions")

    path = "C:/projeler/fly_op/results/activation_3d_demo.json"
    with open(path, "r", encoding="utf-8") as f:
        out = json.load(f)

    out["backdrop_positions"] = backdrop
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote backdrop into {path}")


if __name__ == "__main__":
    main()
