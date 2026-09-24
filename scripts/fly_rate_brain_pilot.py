"""Pilot: does a differentiable leaky-rate substrate (rate_brain.py,
inspired by cesp99/fly-chess's architecture, reviewed 2026-09-17) show a
real-connectome-vs-null-model gap that the LIF spiking substrate never did
(Faz 1/1b/1c/S1: flywire tied or lost to er_null every time)? Small-scale
pilot per the user's request ("kucuk olcekte once dene kur") -- an induced
~3,000-neuron subgraph, not the full connectome. Informal, not
pre-registered, single seed: a feasibility check before any bigger
commitment.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from flyopt.benchmarks import RASTRIGIN_BOUNDS
from flyopt.substrates.graph_builders import er_null
from flyopt.variants.rate_brain import RateBrain, RateBrainConfig, build_subgraph_bfs, build_training_set, train

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DIM = 2
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 400
LR = 3e-3


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    rng = np.random.default_rng(7000)
    encode_pick = rng.choice(afferent, size=N_RAYS, replace=False)
    decode_pick = rng.choice(efferent, size=N_READOUT, replace=False)
    must_include = np.concatenate([encode_pick, decode_pick])

    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(
        base_weights, encode_pick, decode_pick, SUBGRAPH_SIZE, seed=7000,
    )
    print(f"real subgraph (BFS from encode neurons): {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    from scipy.sparse.csgraph import shortest_path
    adj = (sub_real != 0).astype(np.int8)
    dist = shortest_path(adj, directed=True, unweighted=True, indices=encode_idx)
    reach = np.isfinite(dist[:, decode_idx]).mean()
    print(f"encode->decode reachability in real subgraph: {reach:.2%}")

    sub_null = er_null(sub_real, seed=7000)
    print(f"er_null subgraph: {sub_null.shape[0]} nodes, {sub_null.nnz} edges (same shape/edge count)")

    X, Y = build_training_set(DIM, RASTRIGIN_BOUNDS, n_problems=25, n_starts=12, ray_radius=0.3, seed=8000)
    print(f"training set: {X.shape[0]} samples")

    cfg = RateBrainConfig(dim=DIM, n_readout=N_READOUT, T=8, ray_radius=0.3, decode_scale=0.5, train_gain=True)

    results = {}
    for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
        print(f"\n--- training on {name} ---")
        brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=42)
        curve = train(brain, X, Y, epochs=EPOCHS, lr=LR)
        results[name] = curve[-1]
        print(f"{name}: final train mse = {curve[-1]:.4f}")

    print("\n=== SONUC ===")
    print(f"real_connectome final mse: {results['real_connectome']:.4f}")
    print(f"er_null         final mse: {results['er_null']:.4f}")
    diff_pct = (results['er_null'] - results['real_connectome']) / results['er_null'] * 100
    print(f"fark: gercek connectome er_null'dan {diff_pct:+.1f}% farkli (pozitif = gercek daha iyi/dusuk mse)")

    import json
    with open("C:/projeler/fly_op/results/fly_rate_brain_pilot.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "diff_pct": diff_pct, "subgraph_size": SUBGRAPH_SIZE,
                    "epochs": EPOCHS, "n_samples": X.shape[0]}, f, indent=2)
    print("wrote results/fly_rate_brain_pilot.json")


if __name__ == "__main__":
    main()
