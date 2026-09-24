"""Extended subgraph-selection sweep (2026-09-21). Follows
fly_slope_stability_subgraphseed_robustness.py's finding: two alternative
subgraph seeds (9001: delta=-0.219, 9002: delta=0.250, both n=8, both
gate FAIL) came nowhere near the official seed=9000's delta=0.884 (n=30).
This raises the central open question: is seed=9000 a genuine, typical
property of female FlyWire FAFB, or an outlier -- the one subgraph draw
that happened to look strong?

Breadth over depth: 8 NEW subgraph-selection seeds (9003-9010), n=4 each
(real vs degree_preserving_rewire) rather than n=8/n=30 -- the question
here is the DISTRIBUTION of delta across many independent subgraph draws,
not a precise estimate for any one of them. Same architecture, task,
training recipe, primary null as every other official test this session.

Combined with the existing 9000/9001/9002 data points, this gives 11
independent subgraph draws total to characterize how typical/atypical
the official finding's magnitude is.
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 2026-09-21: 13.8x speedup measured on this machine's RTX 4060
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(4))  # n=4 per subgraph -- breadth over depth for this sweep
SUBGRAPH_SEEDS = list(range(9003, 9011))  # 8 new subgraph draws


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

    log_path = f"C:/projeler/fly_op/results/subgraphseed_sweep_{subgraph_seed}.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            for name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test, device=DEVICE))
                    test_mse = float(((pred - torch.as_tensor(Y_test, device=DEVICE)) ** 2).mean().item())
                results[name].append(test_mse)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": test_mse}) + "\n")
                f.flush()
                print(f"  subgraph_seed={subgraph_seed} seed={seed} [{name:24s}] test_mse={test_mse:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["degree_preserving_null"])
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    try:
        w_stat, w_p = stats.wilcoxon(real, null)
    except ValueError:
        w_p = float("nan")
    try:
        mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    except ValueError:
        mw_p = float("nan")

    print(f"\n=== subgraph_seed={subgraph_seed} (n={len(SEEDS)}): real median={np.median(real):.4f} "
          f"null median={np.median(null):.4f} delta={cliffs_delta:.3f} mannwhitney_p={mw_p:.4f} ===\n")
    return {"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
            "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta}


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    all_results = {}
    for subgraph_seed in SUBGRAPH_SEEDS:
        all_results[subgraph_seed] = run_for_subgraph_seed(base_weights, subgraph_seed)

    with open("C:/projeler/fly_op/results/subgraphseed_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    deltas = [all_results[s]["cliffs_delta"] for s in SUBGRAPH_SEEDS]
    print("\n=== TUM SUBGRAF TOHUMLARI OZET (breadth sweep, n=4 her biri) ===")
    for s in SUBGRAPH_SEEDS:
        print(f"  seed={s}: delta={all_results[s]['cliffs_delta']:.3f}")
    print(f"\nreferans (n farkli): seed=9000 delta=0.884 (n=30) | seed=9001 delta=-0.219 (n=8) | seed=9002 delta=0.250 (n=8)")
    print(f"bu sweep (n=4): medyan delta={np.median(deltas):.3f}  min={min(deltas):.3f}  max={max(deltas):.3f}")
    print("wrote results/subgraphseed_sweep_summary.json")


if __name__ == "__main__":
    main()
