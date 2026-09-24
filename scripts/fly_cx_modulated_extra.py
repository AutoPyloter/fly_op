"""CX-modulated search vs classic annealed search vs ER-null (2026-09-18,
idea #4: let the fly's own central-complex activity set the
exploration/exploitation balance instead of a hand-scheduled anneal).
8 seeds, same Rastrigin multi-problem task as the other pilots.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse, stats

from flyopt.substrates.graph_builders import er_null
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
SEEDS = list(range(8, 16))


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    cx_pool = np.load(f"{CELLTYPES}/central_complex_indices.npy")

    # Simplification (2026-09-18): afferent -> efferent needed >6 hops through
    # these candidates (0 reachable in testing) while afferent -> CX was easy
    # (CX gets heavy sensory integration input, biologically expected) -- so
    # BOTH readouts (direction and gain) come from CX itself, split into two
    # non-overlapping halves, instead of requiring a second, harder-to-reach
    # motor pool.
    encode_full, cx_verified = select_connected_encode_decode(
        base_weights, afferent, cx_pool, n_encode=N_RAYS, n_decode=N_DECODE + N_CX,
        max_hops=6, n_encode_candidates=200, seed=16000,
    )
    print(f"verified: {len(encode_full)} encode, {len(cx_verified)} CX (total, split into decode/gain halves)")
    if len(cx_verified) < N_DECODE + N_CX:
        raise RuntimeError(f"only {len(cx_verified)} CX neurons reachable, need {N_DECODE + N_CX} -- widen the search")
    decode_full = cx_verified[:N_DECODE]
    cx_full = cx_verified[N_DECODE:N_DECODE + N_CX]

    all_targets = np.concatenate([decode_full, cx_full])
    sub_real, encode_idx, mapped_targets, _nodes = build_subgraph_bfs(base_weights, encode_full, all_targets, SUBGRAPH_SIZE, seed=16000)
    decode_idx = mapped_targets[: len(decode_full)]
    cx_idx = mapped_targets[len(decode_full):]
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = CXModulatedConfig(dim=2, T_inner=4, episode_steps=20, ray_radius=0.3)
    log_path = "C:/projeler/fly_op/results/fly_cx_modulated_multiseed_extra.jsonl"
    results = {"real_connectome": [], "er_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = er_null(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                brain = CXModulatedBrain(sub, encode_idx, decode_idx, cx_idx, cfg, seed=seed)
                train_cx_modulated(brain, epochs=EPOCHS, lr=LR, seed=seed, geometries_per_step=6)
                eval_fx = evaluate_cx_modulated(brain, n_problems=20, seed=seed + 6000)
                results[name].append(eval_fx)
                f.write(json.dumps({"seed": seed, "substrate": name, "eval_fx": eval_fx}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] eval_fx={eval_fx:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["er_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (CX-modulated Rastrigin arama, n=8 tohum) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_cx_modulated_summary_extra.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_cx_modulated_summary_extra.json")


if __name__ == "__main__":
    main()
