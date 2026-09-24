"""Subgraph-selection robustness check (2026-09-21). EVERY experiment this
session (slope stability delta=0.884 n=30, beam design delta=0.613 n=30,
the 3D visualization) has used the exact SAME encode/decode selection seed
(9000) -- meaning the same one arbitrarily-picked 3000-node subgraph of
FlyWire FAFB every time. This has never been varied, which leaves an
unexamined confound: is the advantage a property of the FEMALE FLYWIRE
CONNECTOME, or an artifact of this ONE particular subgraph pick?

Tests 2 fresh encode/decode selection seeds (9001, 9002 -- same
select_connected_encode_decode/build_subgraph_bfs procedure, same
SUBGRAPH_SIZE, just a different random starting point for which afferent/
efferent candidates get used) on the official slope-stability task against
degree_preserving_rewire, n=8 each. If delta stays positive/large across
different arbitrary subgraph picks, the finding is a property of the
connectome, not a cherry-picked subgraph. If it collapses, the official
finding's scope must be narrowed to "this one subgraph", not "FlyWire".
"""
from __future__ import annotations

import json

import numpy as np
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

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(8))
SUBGRAPH_SEEDS = [9001, 9002]  # 9000 is the baseline used everywhere else (delta=0.884, n=30)


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


def run_for_subgraph_seed(base_weights, subgraph_seed: int) -> dict:
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=subgraph_seed,
    )
    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(
        base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=subgraph_seed
    )
    print(f"subgraph_seed={subgraph_seed}: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, "
          f"{len(encode_idx)} encode / {len(decode_idx)} decode")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = f"C:/projeler/fly_op/results/subgraphseed_robustness_{subgraph_seed}.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            for name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test))
                    test_mse = float(((pred - torch.as_tensor(Y_test)) ** 2).mean().item())
                results[name].append(test_mse)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": test_mse}) + "\n")
                f.flush()
                print(f"  subgraph_seed={subgraph_seed} seed={seed} [{name:24s}] test_mse={test_mse:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["degree_preserving_null"])
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33
    print(f"\n=== subgraph_seed={subgraph_seed}: real median={np.median(real):.4f} "
          f"null median={np.median(null):.4f} Wilcoxon p={w_p:.5f} Mann-Whitney p={mw_p:.5f} "
          f"delta={cliffs_delta:.3f} gate={'PASS' if gate else 'FAIL'} ===\n")
    return {"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
            "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    all_results = {}
    for subgraph_seed in SUBGRAPH_SEEDS:
        all_results[subgraph_seed] = run_for_subgraph_seed(base_weights, subgraph_seed)

    with open("C:/projeler/fly_op/results/subgraphseed_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("wrote results/subgraphseed_robustness_summary.json")
    print("\n(baseline for comparison: subgraph_seed=9000, delta=0.884, n=30, Mann-Whitney p~=0.000000)")


if __name__ == "__main__":
    main()
