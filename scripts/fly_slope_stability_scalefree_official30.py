"""Extend fly_slope_stability_scalefree_multiseed.py to n=30 (2026-09-20),
matching the same rigor as the official degree_preserving_rewire result
(delta=0.884, n=30, Mann-Whitney p~=0.000000). Current state: n=8,
delta=0.844, Mann-Whitney p=0.00295 -- already PASSes the magnitude+
significance gate at n=8, but per the session's standard, extending to
n=30 for the same confidence level as the primary official result.

Runs 22 NEW seeds (8-29), combines with the existing n=8
(fly_slope_stability_scalefree_summary.json) for n=30.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import scale_free_null
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
    train,
)

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(8, 30))


def build_training_set_geo(n_problems: int, n_starts: int, seed: int):
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
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    all_nodes = np.arange(SUBGRAPH_SIZE)

    log_path = "C:/projeler/fly_op/results/fly_slope_stability_scalefree_seeds8to29.jsonl"
    results = {"real_connectome": [], "scale_free_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            sub_synth = scale_free_null(sub_real, seed=seed)

            synth_encode_full, synth_decode_full = select_connected_encode_decode(
                sub_synth, all_nodes, all_nodes, n_encode=N_RAYS, n_decode=N_READOUT,
                max_hops=6, n_encode_candidates=200, seed=20000 + seed,
            )
            coo = sub_synth.tocoo()
            print(f"seed={seed} scale_free_null: {sub_synth.shape[0]} nodes, {coo.nnz} edges, "
                  f"{len(synth_encode_full)} verified encode / {len(synth_decode_full)} verified decode")

            for name, sub, e_idx, d_idx in [
                ("real_connectome", sub_real, encode_idx, decode_idx),
                ("scale_free_null", sub_synth, synth_encode_full, synth_decode_full[:N_READOUT]),
            ]:
                if len(d_idx) < 5:
                    raise RuntimeError(f"seed={seed} {name}: only {len(d_idx)} decode neurons reachable")
                cfg = RateBrainConfig(dim=DIM, n_readout=len(d_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)
                brain = RateBrain(sub, e_idx, d_idx, cfg, seed=seed)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)

                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test))
                    test_mse = float(((pred - torch.as_tensor(Y_test)) ** 2).mean().item())

                results[name].append(test_mse)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": test_mse}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] test_mse={test_mse:.4f}")

    with open("C:/projeler/fly_op/results/fly_slope_stability_scalefree_seeds8to29_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("C:/projeler/fly_op/results/fly_slope_stability_scalefree_summary.json") as f:
        orig = json.load(f)
    combined_real = np.array(orig["real"] + results["real_connectome"])
    combined_null = np.array(orig["null"] + results["scale_free_null"])
    assert len(combined_real) == 30 and len(combined_null) == 30, f"expected n=30, got {len(combined_real)}/{len(combined_null)}"

    mw_stat, mw_pvalue = stats.mannwhitneyu(combined_real, combined_null, alternative="two-sided")
    w_stat, w_pvalue = stats.wilcoxon(combined_real, combined_null)
    gt = np.sum(combined_real[:, None] < combined_null[None, :])
    lt = np.sum(combined_real[:, None] > combined_null[None, :])
    cliffs_delta = float((gt - lt) / (len(combined_real) * len(combined_null)))

    print("\n=== SEV STABILITESI, SENTETIK HUB-BASKIN (SCALE-FREE) NULL, n=30 ===")
    print(f"real median={np.median(combined_real):.4f}  null median={np.median(combined_null):.4f}")
    print(f"Mann-Whitney U: p={mw_pvalue:.6f}")
    print(f"Wilcoxon signed-rank: p={w_pvalue:.6f}")
    print(f"Cliff's delta: {cliffs_delta:.3f}")
    print(f"gate (Mann-Whitney p<0.05 AND |delta|>0.33): {'PASS' if (mw_pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_slope_stability_scalefree_official30_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "real": combined_real.tolist(), "null": combined_null.tolist(),
            "mannwhitney_p": float(mw_pvalue), "wilcoxon_p": float(w_pvalue), "cliffs_delta": cliffs_delta,
        }, f, indent=2)
    print("wrote results/fly_slope_stability_scalefree_official30_summary.json")


if __name__ == "__main__":
    main()
