"""Compare 4 swarm organizations (classic gbest PSO, chemotaxis/no-gbest,
lek/multi-attractor, ring-topology lbest PSO) using the SAME trained
CX-modulated fly nudge, same real-vs-null comparison, same 8 seeds.
Trains ONCE per (substrate, seed) and reuses the trained particles across
all 4 topologies (same initial swarm state too) so the topology
comparison isn't confounded by re-training noise.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks import RASTRIGIN_BOUNDS, rastrigin
from flyopt.substrates.graph_builders import er_null
from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode
from flyopt.variants.cx_modulated_search import CXModulatedBrain, CXModulatedConfig, train_cx_modulated
from flyopt.variants.swarm_topologies import chemotaxis_step, lek_pull, ring_lbest
from flyopt.variants.fly_proposer_scene import _rays

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
CELLTYPES = f"{DATA_PROCESSED}/celltypes"
DIM = 2
N_RAYS = 4
N_DECODE = 20
N_CX = 15
SUBGRAPH_SIZE = 2000
N_PARTICLES = 12
N_ITERS = 60
W, C1, C2 = 0.6, 1.2, 1.2
LEK_TOP_K = 3
SEEDS = list(range(8))
TRAIN_EPOCHS = 100
TOPOLOGIES = ["gbest", "chemotaxis", "lek", "ring_lbest"]


def build_swarm_brains(sub, encode_idx, decode_idx, cx_idx, cfg, trained_state):
    flies = [CXModulatedBrain(sub, encode_idx, decode_idx, cx_idx, cfg, seed=1000 + i) for i in range(N_PARTICLES)]
    for fly in flies:
        for k, v in trained_state.items():
            getattr(fly, k).data.copy_(v)
        fly.h_state = torch.zeros(fly.n, 1)
    return flies


def fly_nudge(brain, x_i, fx_i):
    rays_np = _rays(x_i, rastrigin, fx_i, 0.3)
    with torch.no_grad():
        w_eff = brain.effective_weights()
        a = torch.sigmoid(brain.leak_logit).unsqueeze(1)
        bias = brain.bias.unsqueeze(1)
        inject = torch.zeros(brain.n, 1)
        inject.index_add_(0, brain.encode_idx, torch.as_tensor(rays_np, dtype=torch.float32).unsqueeze(1))
        h = brain.h_state
        for _ in range(brain.config.T_inner):
            h = brain._substep(h, w_eff, a, bias, inject)
        brain.h_state = h
        decode_h = h.index_select(0, brain.decode_idx).squeeze(1)
        pred_dir = (brain.readout_dir @ decode_h).numpy()
        cx_h = h.index_select(0, brain.cx_idx).squeeze(1)
        gain = float(torch.sigmoid(brain.readout_gain @ cx_h + brain.readout_gain_bias).item())
    return gain * pred_dir


def run_swarm_topology(topology, sub, encode_idx, decode_idx, cx_idx, cfg, trained_state, x0, v0):
    lo, hi = RASTRIGIN_BOUNDS
    rng = np.random.default_rng(0)  # only used for gbest/ring_lbest's r1/r2 draws
    flies = build_swarm_brains(sub, encode_idx, decode_idx, cx_idx, cfg, trained_state)
    x, v = x0.copy(), v0.copy()
    fx = np.array([rastrigin(xi) for xi in x])
    pbest_x, pbest_fx = x.copy(), fx.copy()
    gbest_fx = float(pbest_fx.min())

    for it in range(N_ITERS):
        r1, r2 = rng.random((N_PARTICLES, DIM)), rng.random((N_PARTICLES, DIM))
        if topology == "gbest":
            gbest_i = int(np.argmin(pbest_fx))
            v = W * v + C1 * r1 * (pbest_x - x) + C2 * r2 * (pbest_x[gbest_i] - x)
        elif topology == "ring_lbest":
            targets = np.array([ring_lbest(pbest_x, pbest_fx, i) for i in range(N_PARTICLES)])
            v = W * v + C1 * r1 * (pbest_x - x) + C2 * r2 * (targets - x)
        elif topology == "chemotaxis":
            pulls = np.array([chemotaxis_step(x, fx, i) for i in range(N_PARTICLES)])
            v = W * v + C2 * r2 * pulls  # no pbest/gbest term at all -- purely decentralized
        elif topology == "lek":
            pulls = np.array([lek_pull(x, fx, i, LEK_TOP_K) for i in range(N_PARTICLES)])
            v = W * v + C1 * r1 * (pbest_x - x) + C2 * r2 * pulls
        else:
            raise ValueError(topology)

        for i in range(N_PARTICLES):
            x[i] = np.clip(x[i] + v[i] + fly_nudge(flies[i], x[i], fx[i]), lo, hi)

        fx = np.array([rastrigin(xi) for xi in x])
        improved = fx < pbest_fx
        pbest_x[improved] = x[improved]
        pbest_fx[improved] = fx[improved]
        gbest_fx = min(gbest_fx, float(pbest_fx.min()))

    return gbest_fx


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    cx_pool = np.load(f"{CELLTYPES}/central_complex_indices.npy")

    encode_full, cx_verified = select_connected_encode_decode(
        base_weights, afferent, cx_pool, n_encode=N_RAYS, n_decode=N_DECODE + N_CX,
        max_hops=6, n_encode_candidates=200, seed=18000,
    )
    decode_full = cx_verified[:N_DECODE]
    cx_full = cx_verified[N_DECODE:N_DECODE + N_CX]
    all_targets = np.concatenate([decode_full, cx_full])
    sub_real, encode_idx, mapped_targets, _nodes = build_subgraph_bfs(base_weights, encode_full, all_targets, SUBGRAPH_SIZE, seed=18000)
    decode_idx = mapped_targets[: len(decode_full)]
    cx_idx = mapped_targets[len(decode_full):]
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = CXModulatedConfig(dim=DIM, T_inner=4, episode_steps=N_ITERS, ray_radius=0.3)
    # see fly_cx_pso_swarm_multiseed.py 2026-09-18 note: training must use a
    # short, proven-stable episode horizon (20), NOT the 60-step swarm
    # horizon, or backprop through 240 sequential recurrent steps overflows
    # to NaN regardless of grad clipping.
    train_cfg = CXModulatedConfig(dim=DIM, T_inner=4, episode_steps=20, ray_radius=0.3)
    log_path = "C:/projeler/fly_op/results/fly_swarm_topologies_multiseed.jsonl"
    results = {t: {"real_connectome": [], "er_null": []} for t in TOPOLOGIES}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = er_null(sub_real, seed=seed)
            rng = np.random.default_rng(seed)
            lo, hi = RASTRIGIN_BOUNDS
            x0 = rng.uniform(lo, hi, size=(N_PARTICLES, DIM))
            v0 = rng.uniform(-1, 1, size=(N_PARTICLES, DIM))

            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                template = CXModulatedBrain(sub, encode_idx, decode_idx, cx_idx, train_cfg, seed=seed)
                train_cx_modulated(template, epochs=TRAIN_EPOCHS, lr=5e-3, seed=seed, geometries_per_step=6)
                trained_state = {k: getattr(template, k).data.clone()
                                  for k in ["readout_dir", "readout_gain", "readout_gain_bias", "gain", "bias", "leak_logit"]}

                for topology in TOPOLOGIES:
                    gbest_fx = run_swarm_topology(topology, sub, encode_idx, decode_idx, cx_idx, cfg, trained_state, x0, v0)
                    results[topology][name].append(gbest_fx)
                    f.write(json.dumps({"seed": seed, "substrate": name, "topology": topology, "gbest_fx": gbest_fx}) + "\n")
                    f.flush()
                    print(f"seed={seed} [{name:16s}] {topology:12s} gbest_fx={gbest_fx:.4f}")

    print("\n\n=== TOPOLOJI OZETI ===")
    summary = {}
    for topology in TOPOLOGIES:
        real = np.array(results[topology]["real_connectome"])
        null = np.array(results[topology]["er_null"])
        wstat, pvalue = stats.wilcoxon(real, null)
        gt = np.sum(real[:, None] < null[None, :])
        lt = np.sum(real[:, None] > null[None, :])
        cliffs_delta = float((gt - lt) / (len(real) * len(null)))
        summary[topology] = {"p": pvalue, "delta": cliffs_delta, "real_median": float(np.median(real)), "null_median": float(np.median(null))}
        print(f"{topology:12s} p={pvalue:.4f}  delta={cliffs_delta:+.3f}  "
              f"real_med={np.median(real):.3f}  null_med={np.median(null):.3f}  "
              f"gate={'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_swarm_topologies_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/fly_swarm_topologies_summary.json")


if __name__ == "__main__":
    main()
