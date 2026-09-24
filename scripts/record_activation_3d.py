"""Record a real slope-stability episode's neuron activations over time,
with real 3D FlyWire positions, for a 3D visualization (2026-09-21, user
request: "sinek beyni optimizasyon yaparken beyninin nereleri
ateslendigini parametrik olarak modelle, 3d bir sekilde gormek
istiyorum"). Trains one real-connectome brain (same recipe as
fly_slope_stability_degreenull_multiseed.py), then runs one representative
episode with gradients off, recording the hidden state `h` after every
T_inner substep of every episode step. Exports a compact JSON: node
positions (real FlyWire pos_x/y/z for the exact 3000 nodes used),
role per node (encode/decode/other), and the activation time series
(quantized to keep file size small).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry, sample_valid_point
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
    train,
)

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DATA_RAW = "C:/projeler/fly_op/data/raw"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEED = 0
EPISODE_STEPS = 12  # steps of the recorded demo episode (kept short for a readable animation)


def build_training_set_geo(n_problems, n_starts, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = random_geometry(rng)
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = sample_valid_point(geo, rng)
            fx = f(x)
            rays = _rays(x, f, fx, RAY_RADIUS)
            target = _teacher_delta(x, rays, RAY_RADIUS)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(DIM)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    root_ids = np.load(f"{DATA_RAW}/proofread_root_ids_783.npy")  # matrix index -> root_id

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    # nodes: array of ORIGINAL full-graph indices for each of the 3000 subgraph rows
    node_root_ids = root_ids[nodes]

    print("loading FlyWire 3D positions...")
    ann = pd.read_csv(f"{DATA_RAW}/Supplemental_file1_neuron_annotations.tsv", sep="\t",
                       usecols=["root_id", "pos_x", "pos_y", "pos_z"])
    ann = ann.drop_duplicates(subset="root_id").set_index("root_id")
    pos = np.full((len(node_root_ids), 3), np.nan, dtype=np.float64)
    found = 0
    for i, rid in enumerate(node_root_ids):
        if int(rid) in ann.index:
            row = ann.loc[int(rid)]
            pos[i] = [row["pos_x"], row["pos_y"], row["pos_z"]]
            found += 1
    print(f"positions found for {found}/{len(node_root_ids)} nodes")
    # fill any missing positions with the centroid of found ones (keeps the point cloud intact)
    if found < len(node_root_ids):
        centroid = np.nanmean(pos, axis=0)
        missing = np.isnan(pos).any(axis=1)
        pos[missing] = centroid
        print(f"filled {missing.sum()} missing positions with centroid")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    print("training real-connectome brain...")
    X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + SEED)
    brain = RateBrain(sub_real, encode_idx, decode_idx, cfg, seed=SEED)
    train(brain, X, Y, epochs=EPOCHS, lr=LR)
    print("training done, recording a demo episode...")

    # one representative demo geometry/trajectory
    rng = np.random.default_rng(777)
    geo = random_geometry(rng)
    f = lambda p, geo=geo: factor_of_safety(p, geo)
    x = sample_valid_point(geo, rng)

    role = np.zeros(sub_real.shape[0], dtype=np.int64)  # 0=other, 1=encode, 2=decode
    role[encode_idx] = 1
    role[decode_idx] = 2

    frames = []  # each frame: list of h-values (one per node) after one substep
    with torch.no_grad():
        for step in range(EPISODE_STEPS):
            fx = f(x)
            rays = _rays(x, f, fx, RAY_RADIUS)
            rays_t = torch.as_tensor(rays, dtype=torch.float32).unsqueeze(0)  # (1, n_encode)

            # RateBrain.forward() starts h=0 fresh every call (no persistent
            # state across steps) and runs T substeps internally in one
            # batched shot -- replicated here substep-by-substep so each
            # intermediate h can be recorded for the animation.
            w = brain.effective_weights()
            a = torch.sigmoid(brain.leak_logit).unsqueeze(1)
            bias = brain.bias.unsqueeze(1)
            h = torch.zeros(brain.n, 1)
            inject = torch.zeros(brain.n, 1)
            inject.index_add_(0, brain.encode_idx, rays_t.t())
            for sub_t in range(cfg.T):
                src = h.index_select(0, brain.col)
                messages = src * w.unsqueeze(1)
                pre = torch.zeros(brain.n, 1).index_add_(0, brain.row, messages)
                pre = pre + inject
                h = (1 - a) * h + a * torch.tanh(pre + bias)
                frames.append(h.squeeze(1).numpy().astype(np.float16).tolist())

            decode_h = h.index_select(0, brain.decode_idx)
            pred_dir = (brain.readout_W @ decode_h).squeeze(1).numpy() * cfg.decode_scale
            x = x + pred_dir
            print(f"step={step} fx={fx:.3f} pred_dir={pred_dir}")

    out = {
        "n_nodes": int(sub_real.shape[0]),
        "positions": pos.tolist(),
        "role": role.tolist(),
        "n_substeps_per_step": cfg.T,
        "episode_steps": EPISODE_STEPS,
        "frames": frames,  # length = episode_steps * T, each a list of n_nodes floats
    }
    with open("C:/projeler/fly_op/results/activation_3d_demo.json", "w", encoding="utf-8") as fjson:
        json.dump(out, fjson)
    print(f"wrote results/activation_3d_demo.json ({len(frames)} frames, {sub_real.shape[0]} nodes)")


if __name__ == "__main__":
    main()
