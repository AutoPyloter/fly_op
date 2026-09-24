"""Rate-brain pilot v2 (2026-09-17, "duzelt ve yeniden dene"): fixes v1's
methodological gap -- encode/decode neurons are no longer picked by
uninformed `rng.choice` from the afferent/efferent pools (which can land
on a pair with zero real forward path, as v1's diagnostic found: half the
encode neurons had zero reachable decode targets). Instead
`select_connected_encode_decode` verifies forward connectivity on the
FULL graph before anything is built. Same real-vs-ER-null comparison,
same differentiable leaky-rate substrate.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

from flyopt.benchmarks import RASTRIGIN_BOUNDS
from flyopt.substrates.graph_builders import er_null
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    build_training_set,
    select_connected_encode_decode,
    train,
)

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DIM = 2
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 400
LR = 3e-3


def check_reachability(sub_csr, encode_idx, decode_idx, label):
    adj_forward = (sub_csr != 0).T.tocsr()  # weights[post,pre] -> transpose gives true pre->post edges
    dist = shortest_path(adj_forward, directed=True, unweighted=True, indices=encode_idx)
    reach = np.isfinite(dist[:, decode_idx])
    print(f"{label}: encode->decode reachable pairs = {reach.mean():.1%}  "
          f"(per-encode-neuron reach counts: {reach.sum(axis=1).tolist()})")


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_idx_full, decode_idx_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=7000,
    )
    print(f"verified-connected selection: {len(encode_idx_full)} encode, {len(decode_idx_full)} decode neurons")
    check_reachability(base_weights, encode_idx_full, decode_idx_full, "full graph (139k nodes)")

    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(
        base_weights, encode_idx_full, decode_idx_full, SUBGRAPH_SIZE, seed=7000,
    )
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")
    check_reachability(sub_real, encode_idx, decode_idx, "real subgraph (3000 nodes)")

    sub_null = er_null(sub_real, seed=7000)
    check_reachability(sub_null, encode_idx, decode_idx, "er_null subgraph (same shape)")

    X, Y = build_training_set(DIM, RASTRIGIN_BOUNDS, n_problems=25, n_starts=12, ray_radius=0.3, seed=8000)
    print(f"training set: {X.shape[0]} samples")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=0.3, decode_scale=0.5, train_gain=True)

    results = {}
    for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
        print(f"\n--- training on {name} ---")
        brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=42)
        curve = train(brain, X, Y, epochs=EPOCHS, lr=LR)
        results[name] = curve[-1]
        print(f"{name}: final train mse = {curve[-1]:.4f}")

    print("\n=== SONUC (v2, dogrulanmis baglantili encode/decode) ===")
    print(f"real_connectome final mse: {results['real_connectome']:.4f}")
    print(f"er_null         final mse: {results['er_null']:.4f}")
    diff_pct = (results['er_null'] - results['real_connectome']) / results['er_null'] * 100
    print(f"fark: gercek connectome er_null'dan {diff_pct:+.1f}% farkli (pozitif = gercek daha iyi/dusuk mse)")

    import json
    with open("C:/projeler/fly_op/results/fly_rate_brain_pilot_v2.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "diff_pct": diff_pct, "subgraph_size": SUBGRAPH_SIZE,
                    "epochs": EPOCHS, "n_samples": X.shape[0],
                    "n_encode": len(encode_idx), "n_decode": len(decode_idx)}, f, indent=2)
    print("wrote results/fly_rate_brain_pilot_v2.json")


if __name__ == "__main__":
    main()
