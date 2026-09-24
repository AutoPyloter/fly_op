"""Null-connectome-controlled reconstruction of Gemini's "sensory fly vs
PSO" notebooks (2026-09-22, user: "aynı mantığı null connectome baseline
ile yeniden kurgula"). Those notebooks (Benchmark_Duyu_Motoru.ipynb,
Benchmark_Karsilastirma_Colab.ipynb, Benchmark_Evrimlesmis_Sinek.ipynb,
Mega_Biyolojik_Turnuva.ipynb) fed the TRUE analytic gradient of the
objective directly into the brain as sensory input, then compared against
PSO (a gradient-free method) -- a first-order-vs-zeroth-order comparison
that tells us nothing about whether connectome STRUCTURE matters, since
almost any network (real, null, or random) relaying a true gradient will
beat blind PSO on smooth-ish functions.

This keeps the same gradient-injection agent design but swaps the
comparison axis to this project's actual question: does the REAL male
CNS connectome outperform a degree_preserving_rewire NULL of the exact
same subgraph, everything else (gradient injection, T, architecture,
agent count, step scale) held identical? Both arms are equally UNTRAINED
(random init) -- matching the original notebooks, which never called
train() either; this isolates the connectome's structure as a fixed
random nonlinear relay, not a learned one.

Multiple independent seeds (rewiring seed + swarm init seed together) are
used, per this project's standard statistical convention (Wilcoxon
signed-rank, Mann-Whitney U, Cliff's delta), instead of the original
notebooks' single uncontrolled run.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse, stats

from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.rate_brain import RateBrain, RateBrainConfig, build_subgraph_bfs, select_connected_encode_decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"

DIM = 2
N_ENCODE = 4  # matches Gemini's notebooks (2D signal, no padding needed)
N_DECODE = 4
SUBGRAPH_SIZE = 3000
T = 20  # matches Gemini's "gave it more time to think" setup
NUM_AGENTS = 2000
ITERS = 100
STEP_SCALE = 0.1
SEEDS = list(range(8))  # this project's standard n=8 pilot


def sphere(x):
    return torch.sum(x**2, dim=1)


def rastrigin(x):
    return 10 * x.shape[1] + torch.sum(x**2 - 10 * torch.cos(2 * np.pi * x), dim=1)


def sensory_fly_optimize(func, brain, num_particles, iters, step_scale, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = (torch.rand((num_particles, DIM), device=DEVICE, generator=gen) * 10 - 5)
    x.requires_grad_(True)
    best_obj = func(x).detach()
    for _ in range(iters):
        obj = func(x)
        obj.sum().backward()
        with torch.no_grad():
            grad = x.grad.clone()
            x.grad.zero_()
            sensory = F.normalize(-grad, p=2, dim=1)
            pad = torch.zeros((x.shape[0], N_ENCODE - DIM), device=DEVICE)
            env_input = torch.cat([sensory, pad], dim=1)
            move = brain(env_input)[:, :DIM] * step_scale
            x_new = (x + move).detach()
            new_obj = func(x_new)
            mask = new_obj < best_obj
            x.data[mask] = x_new[mask]
            best_obj[mask] = new_obj[mask]
    return float(best_obj.min().item())


def run_task(task_name, func, sub_real, encode_idx, decode_idx):
    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=T, decode_scale=0.5, train_gain=True)
    real_list, null_list = [], []
    for seed in SEEDS:
        sub_null = degree_preserving_rewire(sub_real, seed=seed)
        for lst, sub in [(real_list, sub_real), (null_list, sub_null)]:
            brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
            final = sensory_fly_optimize(func, brain, NUM_AGENTS, ITERS, STEP_SCALE, seed=seed)
            lst.append(final)
        print(f"  [{task_name}] seed={seed}: real={real_list[-1]:.5f}  null={null_list[-1]:.5f}")

    real, null = np.array(real_list), np.array(null_list)
    w_p = stats.wilcoxon(real, null).pvalue
    mw_p = stats.mannwhitneyu(real, null, alternative="two-sided").pvalue
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    delta = float((gt - lt) / (len(real) * len(null)))
    gate = bool(mw_p < 0.05 and abs(delta) > 0.33)
    print(f"\n=== {task_name}: real_median={np.median(real):.5f} null_median={np.median(null):.5f} "
          f"Wilcoxon p={w_p:.4f} Mann-Whitney p={mw_p:.4f} delta={delta:.3f} gate={'PASS' if gate else 'FAIL'} ===\n")
    return {"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
            "mannwhitney_p": float(mw_p), "cliffs_delta": delta, "gate_pass": gate}


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/malecns_adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/malecns_afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/malecns_efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_ENCODE, n_decode=N_DECODE,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"male CNS subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, "
          f"{len(encode_idx)} encode / {len(decode_idx)} decode")

    results = {}
    results["sphere"] = run_task("sphere", sphere, sub_real, encode_idx, decode_idx)
    results["rastrigin"] = run_task("rastrigin", rastrigin, sub_real, encode_idx, decode_idx)

    with open("C:/projeler/fly_op/results/sensory_fly_nullcontrol_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote results/sensory_fly_nullcontrol_summary.json")


if __name__ == "__main__":
    main()
