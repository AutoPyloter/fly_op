"""Continuation of fly_rayradius_nullfamily_mechanism.py after it crashed
on community_preserving_rewire (missing `communities` arg -- fixed here
by computing Louvain communities on the real subgraph the same way
fly_slope_stability_communitypreserving_official30.py does). Reuses the
already-computed real-network r=0.1 zero-shot MSEs from the first run's
jsonl (same subgraph/encode/decode/training seeds, so the real network
per seed is identical -- no need to retrain) and only runs the two
remaining null conditions: community_preserving and weight_shuffle.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import community_preserving_rewire, weight_shuffle
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
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
TRAIN_RAY_RADIUS = 1.0
EVAL_RADIUS = 0.1
SEEDS = list(range(8))


def build_set(n_problems, n_starts, seed, ray_radius):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = random_geometry(rng)
        lo, hi = geo.param_bounds()
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi)
            fx = f(x)
            rays = _rays(x, f, fx, ray_radius)
            target = _teacher_delta(x, rays, ray_radius)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(DIM)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def compute_real_communities(sub_real: sparse.csr_matrix) -> np.ndarray:
    coo = sub_real.tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(sub_real.shape[0]))
    edges = set((min(r, c), max(r, c)) for r, c in zip(coo.row.tolist(), coo.col.tolist()) if r != c)
    G.add_edges_from(edges)
    communities_sets = nx.algorithms.community.louvain_communities(G, seed=0, resolution=1.0)
    communities = np.zeros(sub_real.shape[0], dtype=np.int64)
    for cid, members in enumerate(communities_sets):
        for m in members:
            communities[m] = cid
    return communities


def gate_stats(real, null):
    real = np.array(real)
    null = np.array(null)
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(delta) > 0.33
    return {"real_median": float(np.median(real)), "null_median": float(np.median(null)),
            "wilcoxon_p": float(w_p), "mannwhitney_p": float(mw_p), "cliffs_delta": delta, "gate_pass": bool(gate)}


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")
    communities = compute_real_communities(sub_real)
    print(f"louvain communities: {communities.max() + 1}")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=TRAIN_RAY_RADIUS, decode_scale=0.5, train_gain=True)

    # reuse already-known real-network r=0.1 zero-shot MSEs (identical seeds/recipe -> identical networks)
    prior = [json.loads(l) for l in open("C:/projeler/fly_op/results/fly_rayradius_nullfamily_mechanism.jsonl", encoding="utf-8")]
    real_by_seed = {r["seed"]: r["real_mse"] for r in prior if r["null"] == "degree_preserving"}
    assert len(real_by_seed) == 8

    null_family = {
        "community_preserving": lambda sub, seed: community_preserving_rewire(sub, communities, seed=seed),
        "weight_shuffle": lambda sub, seed: weight_shuffle(sub, seed=seed),
    }

    summary = {}
    log_path = "C:/projeler/fly_op/results/fly_rayradius_nullfamily_mechanism_part2.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for null_name, null_fn in null_family.items():
            null_mses = []
            for seed in SEEDS:
                X, Y = build_set(n_problems=20, n_starts=10, seed=9500 + seed, ray_radius=TRAIN_RAY_RADIUS)
                sub_null = null_fn(sub_real, seed)
                brain = RateBrain(sub_null, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                X_test, Y_test = build_set(n_problems=10, n_starts=10, seed=19500 + seed, ray_radius=EVAL_RADIUS)
                X_test_t = torch.as_tensor(X_test, device=DEVICE)
                Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
                with torch.no_grad():
                    pred = brain(X_test_t)
                    mse = float(((pred - Y_test_t) ** 2).mean().item())
                null_mses.append(mse)
                f.write(json.dumps({"null": null_name, "seed": seed, "real_mse": real_by_seed[seed], "null_mse": mse}) + "\n")
                f.flush()
                del brain
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print(f"  {null_name} seed={seed} null_mse={mse:.4f}")
            real_ordered = [real_by_seed[s] for s in SEEDS]
            s = gate_stats(real_ordered, null_mses)
            summary[null_name] = s
            print(f"{null_name}: real med={s['real_median']:.4f} null med={s['null_median']:.4f} "
                  f"MW p={s['mannwhitney_p']:.5f} delta={s['cliffs_delta']:.3f} gate={'PASS' if s['gate_pass'] else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_rayradius_nullfamily_mechanism_part2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/fly_rayradius_nullfamily_mechanism_part2_summary.json")


if __name__ == "__main__":
    main()
