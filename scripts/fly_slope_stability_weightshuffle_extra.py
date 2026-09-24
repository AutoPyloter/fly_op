"""Extra 8 seeds (8-15) for fly_slope_stability_weightshuffle_multiseed.py
-- n=8 crossed the |delta|>0.33 magnitude threshold (delta=0.375, real
direction) but failed significance (p=0.25). Per the session's
established rule ("n=8 threshold-passing result should not be trusted
without extending to n=16"), extending here -- this is now the pivotal
check for the whole project: slope-stability is the ONLY finding that
has survived degree_preserving_rewire (delta=1.000) after malecns and
CX-modulated both collapsed against it; weight_shuffle is a third,
complementary null (exact same topology, shuffled weight values).
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import weight_shuffle
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
SEEDS = list(range(8, 16))


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

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_slope_stability_weightshuffle_multiseed_extra.jsonl"
    results = {"real_connectome": [], "weight_shuffle_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            sub_null = weight_shuffle(sub_real, seed=seed)

            for name, sub in [("real_connectome", sub_real), ("weight_shuffle_null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)

                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test))
                    test_mse = float(((pred - torch.as_tensor(Y_test)) ** 2).mean().item())

                results[name].append(test_mse)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": test_mse}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:20s}] test_mse={test_mse:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["weight_shuffle_null"])
    print("\n=== SONUC (extra, seeds 8-15) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")

    with open("C:/projeler/fly_op/results/fly_slope_stability_weightshuffle_summary_extra.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist()}, f, indent=2)

    with open("C:/projeler/fly_op/results/fly_slope_stability_weightshuffle_summary.json") as f:
        orig = json.load(f)
    combined_real = np.array(orig["real"] + real.tolist())
    combined_null = np.array(orig["null"] + null.tolist())
    wstat, pvalue = stats.wilcoxon(combined_real, combined_null)
    gt = np.sum(combined_real[:, None] < combined_null[None, :])
    lt = np.sum(combined_real[:, None] > combined_null[None, :])
    cliffs_delta = float((gt - lt) / (len(combined_real) * len(combined_null)))
    print("\n=== BIRLESTIRILMIS SONUC (n=16 tohum) ===")
    print(f"real median={np.median(combined_real):.4f}  null median={np.median(combined_null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_slope_stability_weightshuffle_summary_n16.json", "w", encoding="utf-8") as f:
        json.dump({"real": combined_real.tolist(), "null": combined_null.tolist(),
                    "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_slope_stability_weightshuffle_summary_n16.json")


if __name__ == "__main__":
    main()
