"""Saccadic navigation task (2026-09-22, user/collaborative idea following
the "sinekler serkeş serkeş dolanıyor" observation): real Drosophila
flight consists of straight bouts interrupted by rapid, ballistic
saccadic turns, not continuous path integration. The project's existing
"continuous" multi-step task (episode_steps=20, small per-step
corrections, persistent state across the whole trajectory) failed
(delta=-0.156); the reset-state ablation (h zeroed each step, but still
20 SMALL steps) only partially recovered (delta=+0.125, still FAIL).

This task tests a qualitatively different regime, matching real saccadic
flight more closely: FEW (6) discrete decision points ("saccades"), each
a fresh (state-reset) single-shot decision like the task that already
works officially (delta=0.884), each committing to one LARGE ballistic
leg (step_scale tuned via smoke test -- 3.0, see EXPERIMENTS.md) with NO
continuation/re-adjustment mid-leg. If the connectome's advantage is
specific to "single fresh decision, then commit" (matching how it
actually flies) rather than "continuous integration" (which it doesn't
do), this should recover the single-shot task's strength; if not, this
sharpens the boundary of where the advantage holds.

Uses the SAME slope-stability landscape/architecture as the official
single-shot and continuous-task results for direct comparability.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode
from flyopt.variants.slope_continuous import (
    SlopeContinuousBrain,
    SlopeContinuousConfig,
    evaluate_slope_continuous,
    train_slope_continuous,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DIM = 3
N_ENCODE = 2 * DIM
N_READOUT = 20
SUBGRAPH_SIZE = 1500  # matches v4 / resetstate ablation for direct comparability
EPOCHS = 100
LR = 5e-3
SEEDS = list(range(8))

N_LEGS = 6          # few, discrete saccade decisions (not 20 small steps)
LEG_STEP_SCALE = 3.0  # tuned via smoke test: 1.0 too timid, 6.0 blows past the valid domain


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

    cfg = SlopeContinuousConfig(dim=DIM, T_inner=8, episode_steps=N_LEGS, ray_radius=1.0,
                                 step_scale=LEG_STEP_SCALE, reset_state_each_step=True)
    log_path = "C:/projeler/fly_op/results/fly_saccadic_navigation_multiseed.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = degree_preserving_rewire(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = SlopeContinuousBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train_slope_continuous(brain, n_problems=10, n_episodes_per_problem=3, epochs=EPOCHS, lr=LR, seed=seed, verbose_every=20, geometries_per_step=6)
                eval_fs = evaluate_slope_continuous(brain, n_problems=8, n_episodes_per_problem=3, seed=seed + 5000)
                results[name].append(eval_fs)
                f.write(json.dumps({"seed": seed, "substrate": name, "eval_fs": eval_fs}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:24s}] eval_fs={eval_fs:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["degree_preserving_null"])
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])  # lower FS found = better search
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33

    print("\n=== SONUC (sakkadik navigasyon, N_LEGS=6, degree_preserving null, n=8) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if gate else 'FAIL'}")
    print("\n(referans: tek-atislik resmi bulgu delta=0.884 n=30; kalici-durum surekli v4 delta=-0.156; durum-sifirlama ablasyonu delta=+0.125)")

    with open("C:/projeler/fly_op/results/fly_saccadic_navigation_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_saccadic_navigation_summary.json")


if __name__ == "__main__":
    main()
