"""Properly-controlled multi-sensory-fusion sweep (2026-09-22). The
single-instance lobe-competition result (visual delta=-0.219, chemo
delta=0.216 n=30, mixed delta=0.884 n=30) is SUGGESTIVE of a fusion
effect, but is confounded with plain subgraph-to-subgraph variance --
the subgraph-seed sweep already showed delta ranging -0.25 to 1.00
across independent draws of the SAME (mixed) pool type. One subgraph
per condition cannot distinguish "mixing modalities helps" from "that
one mixed subgraph happened to be a lucky draw".

This sweeps MULTIPLE independent subgraph-selection seeds (breadth over
depth, n=4 real-vs-null per subgraph, matching
fly_slope_stability_subgraphseed_sweep.py's convention) for THREE afferent
pool conditions:
  - mixed:  full afferent pool (all modalities, as used everywhere else)
  - visual: cell_class == "visual" only
  - chemo:  olfactory/mechanosensory/gustatory/... only (non-visual)

If mixed pools are SYSTEMATICALLY stronger across independent draws
(not just the one official seed=9000), that supports a real fusion
effect. If mixed/visual/chemo distributions overlap heavily, the
lobe-competition result was likely just subgraph-seed variance.
"""
from __future__ import annotations

import gc
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import RateBrain, RateBrainConfig, build_subgraph_bfs, select_connected_encode_decode, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DATA_RAW = "C:/projeler/fly_op/data/raw"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(4))  # n=4 per subgraph -- breadth over depth
SUBGRAPH_SEEDS = [9000, 9011]  # 9000 = official/already-known point, +1 fresh independent draw (reduced from 3 -> 2 after OOM kill)

NON_VISUAL_CLASSES = {"olfactory", "mechanosensory", "gustatory", "thermosensory", "hygrosensory", "unknown_sensory"}


def get_pools():
    root_ids = np.load(f"{DATA_RAW}/proofread_root_ids_783.npy")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    afferent_root_ids = root_ids[afferent]
    ann = pd.read_csv(f"{DATA_RAW}/Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=["root_id", "cell_class"])
    ann = ann.drop_duplicates(subset="root_id").set_index("root_id")
    cell_class = ann.reindex(afferent_root_ids)["cell_class"]
    visual_pool = afferent[(cell_class == "visual").to_numpy()]
    chemo_pool = afferent[cell_class.isin(NON_VISUAL_CLASSES).to_numpy()]
    return {"mixed": afferent, "visual": visual_pool, "chemo": chemo_pool}, efferent


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


def run_one(pool_name, afferent_pool, efferent, subgraph_seed, base_weights):
    n_candidates = min(200, len(afferent_pool))
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent_pool, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=n_candidates, seed=subgraph_seed,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=subgraph_seed)
    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    real_list, null_list = [], []
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
            del brain
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  [{pool_name} seed={subgraph_seed}] rep={seed}: real={real_list[-1]:.4f} null={null_list[-1]:.4f}")

    real, null = np.array(real_list), np.array(null_list)
    gt = np.sum(real[:, None] < null[None, :]); lt = np.sum(real[:, None] > null[None, :])
    delta = float((gt - lt) / (len(real) * len(null)))
    print(f"=== [{pool_name} seed={subgraph_seed}] delta={delta:.3f} ===")
    return delta


def main():
    # 2026-09-22: split into one condition per process invocation (CLI arg)
    # after the all-in-one run got OOM-killed by the system while idle.
    # `python sensory_fusion_seedsweep.py mixed|visual|chemo` -- each run
    # writes its own result file; combine them afterward.
    only_condition = sys.argv[1] if len(sys.argv) > 1 else None

    pools, efferent = get_pools()
    print({k: len(v) for k, v in pools.items()})
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")

    conditions = {only_condition: pools[only_condition]} if only_condition else pools

    all_results = {}
    for pool_name, afferent_pool in conditions.items():
        deltas = []
        for subgraph_seed in SUBGRAPH_SEEDS:
            d = run_one(pool_name, afferent_pool, efferent, subgraph_seed, base_weights)
            deltas.append(d)
        all_results[pool_name] = deltas

    suffix = f"_{only_condition}" if only_condition else ""
    with open(f"C:/projeler/fly_op/results/sensory_fusion_seedsweep_summary{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== OZET ({only_condition or 'tum kosullar'}, alt-graf tohumlari={SUBGRAPH_SEEDS}, n=4 her biri) ===")
    for pool_name, deltas in all_results.items():
        print(f"{pool_name:8s}: deltas={[round(d,3) for d in deltas]}  medyan={np.median(deltas):.3f}")
    print(f"wrote results/sensory_fusion_seedsweep_summary{suffix}.json")


if __name__ == "__main__":
    main()
