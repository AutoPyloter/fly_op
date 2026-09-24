"""Extends fly_lobe_competition_multiseed.py's "chemo_mechano" arm to
n=30 (2026-09-22) -- n=8 crossed the magnitude threshold (delta=0.375)
but not significance (p=0.235), the session's standard "promising but
underpowered" pattern that has triggered n=30 extension every other time
(e.g. beam design). Runs 22 NEW seeds (8-29), combines with the existing
n=8 (lobe_competition_chemo_mechano.jsonl) for n=30.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
    train,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DATA_RAW = "C:/projeler/fly_op/data/raw"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(8, 30))

NON_VISUAL_CLASSES = {"olfactory", "mechanosensory", "gustatory", "thermosensory", "hygrosensory", "unknown_sensory"}


def build_training_set_geo(n_problems, n_starts, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = random_geometry(rng)
        lo, hi = geo.param_bounds()
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi)
            fx = f(x)
            rays = _rays(x, f, fx, RAY_RADIUS)
            target = _teacher_delta(x, rays, RAY_RADIUS)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(DIM)
            X.append(rays); Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    root_ids = np.load(f"{DATA_RAW}/proofread_root_ids_783.npy")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    afferent_root_ids = root_ids[afferent]
    ann = pd.read_csv(f"{DATA_RAW}/Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=["root_id", "cell_class"])
    ann = ann.drop_duplicates(subset="root_id").set_index("root_id")
    cell_class = ann.reindex(afferent_root_ids)["cell_class"]
    nonvisual_mask = cell_class.isin(NON_VISUAL_CLASSES).to_numpy()
    chemo_pool = afferent[nonvisual_mask]
    print(f"chemo/mechano afferent pool: {len(chemo_pool)}")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, chemo_pool, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, {len(encode_idx)} encode / {len(decode_idx)} decode")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/lobe_chemo_mechano_seeds8to29.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(20, 10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(10, 10, seed=19500 + seed)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test, device=DEVICE))
                    mse = float(((pred - torch.as_tensor(Y_test, device=DEVICE)) ** 2).mean().item())
                results[name].append(mse)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": mse}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:24s}] test_mse={mse:.4f}")

    with open("C:/projeler/fly_op/results/lobe_chemo_mechano_seeds8to29_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("C:/projeler/fly_op/results/lobe_competition_chemo_mechano.jsonl") as f:
        orig_real, orig_null = [], []
        for line in f:
            d = json.loads(line)
            orig_real.append(d["real"]); orig_null.append(d["null"])

    combined_real = np.array(orig_real + results["real_connectome"])
    combined_null = np.array(orig_null + results["degree_preserving_null"])
    assert len(combined_real) == 30 and len(combined_null) == 30, f"expected n=30, got {len(combined_real)}/{len(combined_null)}"

    mw_stat, mw_p = stats.mannwhitneyu(combined_real, combined_null, alternative="two-sided")
    w_stat, w_p = stats.wilcoxon(combined_real, combined_null)
    gt = np.sum(combined_real[:, None] < combined_null[None, :])
    lt = np.sum(combined_real[:, None] > combined_null[None, :])
    cliffs_delta = float((gt - lt) / (len(combined_real) * len(combined_null)))

    print("\n=== KOKU/DOKUNMA KOKENLI ALT-GRAF, n=30 ===")
    print(f"real medyan={np.median(combined_real):.4f}  null medyan={np.median(combined_null):.4f}")
    print(f"Mann-Whitney p={mw_p:.6f}  Wilcoxon p={w_p:.6f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (mw_p < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/lobe_chemo_mechano_official30_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": combined_real.tolist(), "null": combined_null.tolist(),
                   "mannwhitney_p": float(mw_p), "wilcoxon_p": float(w_p), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/lobe_chemo_mechano_official30_summary.json")


if __name__ == "__main__":
    main()
