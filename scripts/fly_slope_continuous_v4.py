"""Slope stability, CONTINUOUS closed-loop (2026-09-18) -- the last empty
cell of PROTOCOL_V2_TASK_SPECTRUM.md's 2x2 matrix: uzamsal-YAPI +
kapali-dongu, cheaper than malecns. Real connectome vs ER-null, 8 seeds.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse, stats

from flyopt.substrates.graph_builders import er_null
from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode
from flyopt.variants.slope_continuous import (
    SlopeContinuousBrain,
    SlopeContinuousConfig,
    evaluate_slope_continuous,
    train_slope_continuous,
)

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DIM = 3
N_ENCODE = 2 * DIM
N_READOUT = 20
SUBGRAPH_SIZE = 1500
EPOCHS = 100
LR = 5e-3
SEEDS = list(range(8))


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_ENCODE, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=13000,
    )
    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=13000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, {len(encode_idx)} encode / {len(decode_idx)} decode")

    cfg = SlopeContinuousConfig(dim=DIM, T_inner=4, episode_steps=20, ray_radius=1.0, step_scale=1.0)
    log_path = "C:/projeler/fly_op/results/fly_slope_continuous_v4_multiseed.jsonl"
    results = {"real_connectome": [], "er_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = er_null(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                brain = SlopeContinuousBrain(sub, encode_idx, decode_idx, cfg, seed=seed)
                train_slope_continuous(brain, n_problems=10, n_episodes_per_problem=3, epochs=EPOCHS, lr=LR, seed=seed, verbose_every=10, geometries_per_step=6)
                eval_fs = evaluate_slope_continuous(brain, n_problems=8, n_episodes_per_problem=3, seed=seed + 5000)
                results[name].append(eval_fs)
                f.write(json.dumps({"seed": seed, "substrate": name, "eval_fs": eval_fs}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] eval_fs={eval_fs:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["er_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    gt = np.sum(real[:, None] < null[None, :])  # lower FS found = better search (more critical surface found)
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (surekli sev stabilitesi, n=8 tohum) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_slope_continuous_v4_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_slope_continuous_v4_summary.json")


if __name__ == "__main__":
    main()
