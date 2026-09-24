"""Ablation on the continuous/multi-step slope-stability negative result
(2026-09-21, user question: "çok adımlılarda bir sonraki noktaya sanki
ilk defa geliyormuş gibi koysak nasıl ilerler" -- what if we reset the
brain's hidden state at the start of every step instead of letting it
persist across the whole episode?).

fly_slope_continuous_v4.py (persistent state across episode_steps, vs
er_null) was FAIL, delta=-0.156 -- the ONLY structural difference from
the official single-shot PASS (delta=0.884) noted in
PROTOCOL_V2_TASK_SPECTRUM.md's 2x2 matrix is "kapalı-döngü VAR" i.e.
persistent state across a multi-step search. This conflates two
different things though: (a) the multi-step SEARCH process itself
(x moving step by step, errors potentially compounding), and (b)
PERSISTENT brain memory across those steps specifically.

This test isolates (b): same continuous multi-step search (x still
updates step by step, best_fs still tracked across the whole episode),
but `reset_state_each_step=True` zeroes h at the start of every step, so
the brain reads each new position "fresh" like the single-shot task
does. Same subgraph/hyperparameters as v4 for a clean comparison, but
against degree_preserving_rewire (this project's primary null since
2026-09-18) rather than v4's er_null.

If delta comes back positive/large: persistent state was the harmful
ingredient, not multi-step search itself. If still null/negative: the
multi-step search process itself is the problem, independent of memory.
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 2026-09-21: measured 13.8x speedup on this machine's RTX 4060
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

    cfg = SlopeContinuousConfig(dim=DIM, T_inner=4, episode_steps=20, ray_radius=1.0, step_scale=1.0,
                                 reset_state_each_step=True)
    log_path = "C:/projeler/fly_op/results/fly_slope_continuous_resetstate_ablation.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = degree_preserving_rewire(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = SlopeContinuousBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train_slope_continuous(brain, n_problems=10, n_episodes_per_problem=3, epochs=EPOCHS, lr=LR, seed=seed, verbose_every=10, geometries_per_step=6)
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

    print("\n=== SONUC (surekli sev stabilitesi, HER ADIMDA h SIFIRLANIYOR, degree_preserving null, n=8) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if gate else 'FAIL'}")
    print("\n(referans: kalici-durum v4 sonucu, er_null'a karsi: delta=-0.156, FAIL)")

    with open("C:/projeler/fly_op/results/fly_slope_continuous_resetstate_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_slope_continuous_resetstate_ablation_summary.json")


if __name__ == "__main__":
    main()
