"""Cartpole balance: PROTOCOL_V2_TASK_SPECTRUM.md spectrum point #4,
"continuous closed-loop control, cheap" -- fills the gap between static
function tasks (Rastrigin/slope-stability/Fly-Surrogate) and malecns's
expensive full embodied driving. Real connectome vs ER-null, 8 seeds
from the start, same verified-connectivity selection as every other
2026-09-17/18 pilot. Fully end-to-end differentiable (brain + cartpole
physics both smooth), no scripted teacher needed -- the control cost
itself is the training loss.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.substrates.graph_builders import er_null
from flyopt.variants.cartpole_control import CartpoleBrain, CartpoleConfig, sample_init_states, train_cartpole
from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_ENCODE = 4  # x, x_dot, theta, theta_dot
N_READOUT = 20
SUBGRAPH_SIZE = 2000  # smaller than the other pilots: episodes are longer (50 steps x T_inner), keep per-step cost down
EPOCHS = 150
LR = 1e-2
BATCH = 16
SEEDS = list(range(8))


def evaluate(brain: CartpoleBrain, seed: int, n_eval: int = 32) -> float:
    with torch.no_grad():
        init = sample_init_states(n_eval, seed=90000 + seed)
        cost, _ = brain.rollout(init)
        return float(cost.item())


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_ENCODE, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=12000,
    )
    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=12000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, {len(encode_idx)} encode / {len(decode_idx)} decode")

    cfg = CartpoleConfig(T_inner=2, episode_steps=50)
    log_path = "C:/projeler/fly_op/results/fly_cartpole_v2_gradclip_multiseed.jsonl"
    results = {"real_connectome": [], "er_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = er_null(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                brain = CartpoleBrain(sub, encode_idx, decode_idx, cfg, seed=seed)
                train_cartpole(brain, epochs=EPOCHS, lr=LR, batch=BATCH, seed=seed)
                eval_cost = evaluate(brain, seed=seed)
                results[name].append(eval_cost)
                f.write(json.dumps({"seed": seed, "substrate": name, "eval_cost": eval_cost}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] eval_cost={eval_cost:.5f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["er_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    gt = np.sum(real[:, None] < null[None, :])  # lower cost = better balance
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (cartpole denge, n=8 tohum) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.5f}  null median={np.median(null):.5f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f} (pozitif = gercek daha iyi/dusuk maliyet)")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_cartpole_v2_gradclip_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_cartpole_v2_gradclip_summary.json")


if __name__ == "__main__":
    main()
