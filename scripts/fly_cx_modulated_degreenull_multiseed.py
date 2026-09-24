"""Robustness check on the session's cleanest finding (2026-09-18,
fly_cx_modulated_multiseed.py: delta=-0.562, 16/16 seeds, real_connectome
vs er_null) -- same architecture, same subgraph, same training, but
against `degree_preserving_rewire` instead of `er_null`. Degree-preserving
is PROTOCOL.md's actual primary null (Maslov-Sneppen double-edge-swap:
keeps each neuron's in/out-degree exactly, destroys specific wiring) and
is a strictly harder null to beat than ER (which doesn't even preserve
degree), since it's structurally closer to the real graph. If the
CX-modulated result is a genuine substrate-identity effect and not an
artifact of ER's degree distribution being unrealistic, it should survive
here too.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse, stats

from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.cx_modulated_search import CXModulatedBrain, CXModulatedConfig, evaluate_cx_modulated, train_cx_modulated
from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
CELLTYPES = f"{DATA_PROCESSED}/celltypes"
N_RAYS = 4
N_DECODE = 20
N_CX = 15
SUBGRAPH_SIZE = 2000
EPOCHS = 100
LR = 5e-3
SEEDS = list(range(8))


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    cx_pool = np.load(f"{CELLTYPES}/central_complex_indices.npy")

    encode_full, cx_verified = select_connected_encode_decode(
        base_weights, afferent, cx_pool, n_encode=N_RAYS, n_decode=N_DECODE + N_CX,
        max_hops=6, n_encode_candidates=200, seed=16000,
    )
    decode_full = cx_verified[:N_DECODE]
    cx_full = cx_verified[N_DECODE:N_DECODE + N_CX]
    all_targets = np.concatenate([decode_full, cx_full])
    sub_real, encode_idx, mapped_targets, _nodes = build_subgraph_bfs(base_weights, encode_full, all_targets, SUBGRAPH_SIZE, seed=16000)
    decode_idx = mapped_targets[: len(decode_full)]
    cx_idx = mapped_targets[len(decode_full):]
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = CXModulatedConfig(dim=2, T_inner=4, episode_steps=20, ray_radius=0.3)
    log_path = "C:/projeler/fly_op/results/fly_cx_modulated_degreenull_multiseed.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = degree_preserving_rewire(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = CXModulatedBrain(sub, encode_idx, decode_idx, cx_idx, cfg, seed=seed)
                train_cx_modulated(brain, epochs=EPOCHS, lr=LR, seed=seed, geometries_per_step=6)
                eval_fx = evaluate_cx_modulated(brain, n_problems=20, seed=seed + 6000)
                results[name].append(eval_fx)
                f.write(json.dumps({"seed": seed, "substrate": name, "eval_fx": eval_fx}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:24s}] eval_fx={eval_fx:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["degree_preserving_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (CX-modulated, degree-preserving null, n=8 tohum) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_cx_modulated_degreenull_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_cx_modulated_degreenull_summary.json")


if __name__ == "__main__":
    main()
