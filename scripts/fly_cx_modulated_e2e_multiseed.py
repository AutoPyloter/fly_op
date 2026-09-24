"""CX-modulated search with a PROPERLY trained gain readout (2026-09-18
follow-up to the documented limitation in EXPERIMENTS.md: the original
fly_cx_modulated_multiseed.py's `readout_gain`/`readout_gain_bias` never
received gradient at all, since the imitation loss only supervises
`pred_dir` -- gain stayed at its random initialization the whole time.
Its clean 16-seed result (delta=-0.562) was still a genuine, finite
substrate-dependent signal, just not evidence that CX "learns" its
modulatory role.

This script uses `run_episode_e2e`/`train_cx_modulated_e2e`
(cx_modulated_search.py) instead: `x` stays a torch tensor across the
whole rollout and the loss is the actual task objective (mean Rastrigin
value along the trajectory), so `readout_gain` finally gets a real
gradient through its effect on step size. Question: does an actually
-trained CX gain change the real-vs-null picture (n=8 first)?
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse, stats

from flyopt.substrates.graph_builders import er_null
from flyopt.variants.cx_modulated_search import (
    CXModulatedBrain, CXModulatedConfig, evaluate_cx_modulated_e2e, train_cx_modulated_e2e,
)
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
    log_path = "C:/projeler/fly_op/results/fly_cx_modulated_e2e_multiseed.jsonl"
    results = {"real_connectome": [], "er_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = er_null(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                brain = CXModulatedBrain(sub, encode_idx, decode_idx, cx_idx, cfg, seed=seed)
                loss_curve = train_cx_modulated_e2e(brain, epochs=EPOCHS, lr=LR, seed=seed, geometries_per_step=6)
                eval_fx = evaluate_cx_modulated_e2e(brain, n_problems=20, seed=seed + 6000)
                results[name].append(eval_fx)
                f.write(json.dumps({"seed": seed, "substrate": name, "eval_fx": eval_fx,
                                     "final_train_loss": loss_curve[-1]}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] eval_fx={eval_fx:.4f} final_train_loss={loss_curve[-1]:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["er_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (e2e-trained CX-modulated Rastrigin arama, n=8 tohum) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_cx_modulated_e2e_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_cx_modulated_e2e_summary.json")


if __name__ == "__main__":
    main()
