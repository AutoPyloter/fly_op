"""Lobe competition (2026-09-22, collaborative idea): does the origin
anatomical modality of a subgraph's sensory pool change the measured
delta? All prior tests (official finding, subgraph-seed sweep) drew
the encode-neuron candidate pool from ALL afferent types pooled
together (visual + olfactory + mechanosensory + gustatory + ...). This
directly builds two anatomically-restricted variants and runs the
identical official pipeline (same efferent/motor pool, same
degree_preserving_rewire null, same slope-stability task, same
train/eval recipe) on each:

  - "visual": afferent candidates restricted to cell_class == "visual"
    (photoreceptors / optic pathway, n=11390 in our afferent pool)
  - "chemo_mechano": afferent candidates restricted to
    olfactory/mechanosensory/gustatory/thermosensory/hygrosensory
    (n=5593), i.e. everything EXCEPT visual

No new data download needed -- cell_class is already in
Supplemental_file1_neuron_annotations.tsv, cross-referenced against our
existing afferent_indices.npy.
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
SEEDS = list(range(8))

NON_VISUAL_CLASSES = {"olfactory", "mechanosensory", "gustatory", "thermosensory", "hygrosensory", "unknown_sensory"}


def build_lobe_pools():
    root_ids = np.load(f"{DATA_RAW}/proofread_root_ids_783.npy")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    afferent_root_ids = root_ids[afferent]

    ann = pd.read_csv(f"{DATA_RAW}/Supplemental_file1_neuron_annotations.tsv", sep="\t",
                       usecols=["root_id", "cell_class"])
    ann = ann.drop_duplicates(subset="root_id").set_index("root_id")
    cell_class = ann.reindex(afferent_root_ids)["cell_class"]

    visual_mask = (cell_class == "visual").to_numpy()
    nonvisual_mask = cell_class.isin(NON_VISUAL_CLASSES).to_numpy()

    visual_pool = afferent[visual_mask]
    nonvisual_pool = afferent[nonvisual_mask]
    print(f"visual afferent pool: {len(visual_pool)}   chemo/mechano afferent pool: {len(nonvisual_pool)}")
    return visual_pool, nonvisual_pool


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


def run_for_pool(name, afferent_pool, base_weights, efferent):
    n_candidates = min(200, len(afferent_pool))
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent_pool, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=n_candidates, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"[{name}] subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, {len(encode_idx)} encode / {len(decode_idx)} decode")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)
    log_path = f"C:/projeler/fly_op/results/lobe_competition_{name}.jsonl"
    real_list, null_list = [], []
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(20, 10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(10, 10, seed=19500 + seed)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)
            for lst, sub in [(real_list, sub_real), (null_list, sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test, device=DEVICE))
                    mse = float(((pred - torch.as_tensor(Y_test, device=DEVICE)) ** 2).mean().item())
                lst.append(mse)
            f.write(json.dumps({"seed": seed, "real": real_list[-1], "null": null_list[-1]}) + "\n")
            f.flush()
            print(f"  [{name}] seed={seed}: real={real_list[-1]:.4f} null={null_list[-1]:.4f}")

    real, null = np.array(real_list), np.array(null_list)
    w_p = stats.wilcoxon(real, null).pvalue
    mw_p = stats.mannwhitneyu(real, null, alternative="two-sided").pvalue
    gt = np.sum(real[:, None] < null[None, :]); lt = np.sum(real[:, None] > null[None, :])
    delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(delta) > 0.33
    print(f"\n=== [{name}] real_median={np.median(real):.4f} null_median={np.median(null):.4f} "
          f"Wilcoxon p={w_p:.4f} Mann-Whitney p={mw_p:.4f} delta={delta:.3f} gate={'PASS' if gate else 'FAIL'} ===\n")
    return {"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
            "mannwhitney_p": float(mw_p), "cliffs_delta": delta, "gate_pass": bool(gate)}


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    visual_pool, nonvisual_pool = build_lobe_pools()

    results = {}
    results["visual"] = run_for_pool("visual", visual_pool, base_weights, efferent)
    results["chemo_mechano"] = run_for_pool("chemo_mechano", nonvisual_pool, base_weights, efferent)

    with open("C:/projeler/fly_op/results/lobe_competition_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n=== KARSILASTIRMA ===")
    print(f"gorsel (optik) kokenli:      delta={results['visual']['cliffs_delta']:.3f}")
    print(f"koku/dokunma kokenli:        delta={results['chemo_mechano']['cliffs_delta']:.3f}")
    print("(referans: karisik havuz, resmi seed=9000, n=30: delta=0.884)")
    print("wrote results/lobe_competition_summary.json")


if __name__ == "__main__":
    main()
