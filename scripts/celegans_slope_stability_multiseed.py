"""Cross-species generalization test (2026-09-21, following Gemini's
suggestion Section 3.2, scoped to the realistic case -- C. elegans, NOT
"mouse brain" which has no complete synaptic-resolution connectome).
Same exact slope-stability single-shot regression task, same RateBrain
architecture, same training recipe as the FlyWire experiments -- only
the substrate changes, to the Cook et al. 2019 (Nature) C. elegans
hermaphrodite chemical connectome (somatic nervous system: 272 neurons,
sensory->motor, see build_celegans_adjacency.py).

Question: does the modularity-related advantage found on FlyWire
(EXPERIMENTS.md 2026-09-20/21: community_preserving_rewire closes about
half the gap to real, delta=0.429 n=30) show up on a COMPLETELY
different, much smaller, independently-evolved nervous system too? C.
elegans's own network already measured highly modular (Q=0.381) and
clustered (0.329), remarkably close to FlyWire's (Q=0.357, 0.276) --
this tests three null models at once (er_null, degree_preserving,
community_preserving) since the network is small enough to make all
three cheap.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import degree_preserving_rewire, er_null, community_preserving_rewire
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import RateBrain, RateBrainConfig, select_connected_encode_decode, train

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(8))


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


def compute_communities(sub: sparse.csr_matrix) -> np.ndarray:
    coo = sub.tocoo()
    n = sub.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = set((min(r, c), max(r, c)) for r, c in zip(coo.row.tolist(), coo.col.tolist()) if r != c)
    G.add_edges_from(edges)
    comm_sets = nx.algorithms.community.louvain_communities(G, seed=0, resolution=1.0)
    communities = np.zeros(n, dtype=np.int64)
    for cid, members in enumerate(comm_sets):
        for m in members:
            communities[m] = cid
    print(f"C. elegans: {len(comm_sets)} Louvain communities, "
          f"modularity={nx.algorithms.community.modularity(G, comm_sets):.4f}, "
          f"avg_clustering={nx.average_clustering(G):.4f}")
    return communities


def run_one(sub_real, encode_idx, decode_idx, null_kind: str, communities, seed: int) -> tuple[float, float]:
    if null_kind == "er_null":
        sub_null = er_null(sub_real, seed=seed)
    elif null_kind == "degree_preserving":
        sub_null = degree_preserving_rewire(sub_real, seed=seed)
    elif null_kind == "community_preserving":
        sub_null = community_preserving_rewire(sub_real, communities, seed=seed)
    else:
        raise ValueError(null_kind)

    X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
    X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)
    results = {}
    for name, sub in [("real_connectome", sub_real), (null_kind, sub_null)]:
        brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed)
        train(brain, X, Y, epochs=EPOCHS, lr=LR)
        with torch.no_grad():
            pred = brain(torch.as_tensor(X_test))
            test_mse = float(((pred - torch.as_tensor(Y_test)) ** 2).mean().item())
        results[name] = test_mse
    return results["real_connectome"], results[null_kind]


def main():
    adj = sparse.load_npz(f"{DATA_PROCESSED}/celegans_adjacency.npz")
    sensory = np.load(f"{DATA_PROCESSED}/celegans_sensory_indices.npy")
    motor = np.load(f"{DATA_PROCESSED}/celegans_motor_indices.npy")

    encode_idx, decode_idx = select_connected_encode_decode(
        adj, sensory, motor, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=len(sensory), seed=9000,
    )
    print(f"C. elegans: n={adj.shape[0]} edges={adj.nnz}  verified {len(encode_idx)} encode / {len(decode_idx)} decode")
    communities = compute_communities(adj)

    all_results = {}
    for null_kind in ["er_null", "degree_preserving", "community_preserving"]:
        results = {"real_connectome": [], null_kind: []}
        log_path = f"C:/projeler/fly_op/results/celegans_slope_stability_{null_kind}_multiseed.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for seed in SEEDS:
                real_mse, null_mse = run_one(adj, encode_idx, decode_idx, null_kind, communities, seed)
                results["real_connectome"].append(real_mse)
                results[null_kind].append(null_mse)
                f.write(json.dumps({"seed": seed, "real_mse": real_mse, "null_mse": null_mse}) + "\n")
                f.flush()
                print(f"[{null_kind:20s}] seed={seed} real={real_mse:.4f} null={null_mse:.4f}")

        real = np.array(results["real_connectome"])
        null = np.array(results[null_kind])
        w_stat, w_p = stats.wilcoxon(real, null)
        mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
        gt = np.sum(real[:, None] < null[None, :])
        lt = np.sum(real[:, None] > null[None, :])
        cliffs_delta = float((gt - lt) / (len(real) * len(null)))
        gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33
        print(f"\n=== C. elegans vs {null_kind} (n={len(SEEDS)}) ===")
        print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
        print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}  gate={'PASS' if gate else 'FAIL'}\n")
        all_results[null_kind] = {
            "real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
            "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate),
        }

    with open("C:/projeler/fly_op/results/celegans_slope_stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("wrote results/celegans_slope_stability_summary.json")


if __name__ == "__main__":
    main()
