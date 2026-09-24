"""Symmetric propagation-budget (T) robustness check (2026-09-21, user
request, disciplined version of an external suggestion: "erkek sinek
ağının seyrek olduğu için T=8 adımda bilgi-aç kaldığını söylüyorsun,
T=15/20 ile tekrar test et"). To avoid hyperparameter-chasing a negative
result (PROTOCOL.md's explicit rule, already invoked for the C. elegans
negative result), T is doubled (8 -> 16) SYMMETRICALLY on BOTH
connectomes in the SAME run, not just the one that failed:

  - Female FlyWire FAFB (seed=9000): official finding was T=8, delta=0.884
    (n=30). If T=16 breaks or meaningfully changes this known-strong
    result, that alone is important context for interpreting any change
    on the male side.
  - Male CNS (seed=9000): official finding was T=8, delta=0.250, gate FAIL
    (EXPERIMENTS.md 2026-09-21). Structural diagnostic
    (compare_subgraph_structure.py) found male CNS subgraph has less than
    half the density/mean-degree of the female one -- the hypothesis
    being tested is whether a longer propagation budget compensates for
    that sparsity.

n=8 pilot on both, same degree_preserving_rewire null, same task
(slope-stability single-shot regression). Reports both deltas side by
side -- this is a SYMMETRIC controlled comparison, not a search for a T
that "fixes" the male result.
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
SEEDS = list(range(8))
T_NEW = 16  # double the T=8 used everywhere else this session
SUBGRAPH_SELECT_SEED = 9000  # same subgraph used for the official findings on both connectomes


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


def run_connectome(name: str, adj_path: str, afferent_path: str, efferent_path: str) -> dict:
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/{adj_path}")
    afferent = np.load(f"{DATA_PROCESSED}/{afferent_path}")
    efferent = np.load(f"{DATA_PROCESSED}/{efferent_path}")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=SUBGRAPH_SELECT_SEED,
    )
    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(
        base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=SUBGRAPH_SELECT_SEED
    )
    print(f"[{name}] subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, "
          f"{len(encode_idx)} encode / {len(decode_idx)} decode, T={T_NEW}")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=T_NEW, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = f"C:/projeler/fly_op/results/T{T_NEW}_symmetric_{name}.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            for sub_name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test, device=DEVICE))
                    test_mse = float(((pred - torch.as_tensor(Y_test, device=DEVICE)) ** 2).mean().item())
                results[sub_name].append(test_mse)
                f.write(json.dumps({"seed": seed, "substrate": sub_name, "test_mse": test_mse}) + "\n")
                f.flush()
                print(f"  [{name}] seed={seed} [{sub_name:24s}] test_mse={test_mse:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["degree_preserving_null"])
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33
    print(f"\n=== [{name}] T={T_NEW}: real median={np.median(real):.4f} null median={np.median(null):.4f} "
          f"Wilcoxon p={w_p:.5f} Mann-Whitney p={mw_p:.5f} delta={cliffs_delta:.3f} gate={'PASS' if gate else 'FAIL'} ===\n")
    return {"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
            "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}


def main():
    all_results = {}
    all_results["female_flywire"] = run_connectome("female_flywire", "adjacency.npz", "afferent_indices.npy", "efferent_indices.npy")
    all_results["male_cns"] = run_connectome("male_cns", "malecns_adjacency.npz", "malecns_afferent_indices.npy", "malecns_efferent_indices.npy")

    with open(f"C:/projeler/fly_op/results/T{T_NEW}_symmetric_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== KARSILASTIRMA (T=8 referans, T=16 bu kosum) ===")
    print(f"female_flywire: T=8 delta=0.884 (n=30, resmi) -> T=16 delta={all_results['female_flywire']['cliffs_delta']:.3f} (n=8)")
    print(f"male_cns:       T=8 delta=0.250 (n=8, FAIL)    -> T=16 delta={all_results['male_cns']['cliffs_delta']:.3f} (n=8)")
    print(f"wrote results/T{T_NEW}_symmetric_summary.json")


if __name__ == "__main__":
    main()
