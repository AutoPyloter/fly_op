"""CX-modulated search INSIDE a real PSO swarm (2026-09-18, "neden PSO
kullanmiyorsun" -- today's single-agent CX-modulated experiment had no
swarm at all). 12 particles, classic PSO velocity (inertia + cognitive +
social pull towards gbest), each particle additionally nudged by its own
persistent-state fly brain whose CX-derived gain scales the nudge -- same
mechanism as cx_modulated_search.py, now embedded in the swarm loop from
fly_swarm_pso_demo.py. Real connectome vs ER-null, 8 seeds, same Rastrigin
task family.
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
SEEDS = list(range(8))
TRAIN_EPOCHS = 100


def build_swarm_brains(sub, encode_idx, decode_idx, cx_idx, cfg, trained_state=None):
    """2026-09-18 ("egitilmis versiyonu once dene"): `trained_state`, if
    given, is a state_dict from one brain pre-trained on the standard
    multi-problem Rastrigin regression task (train_cx_modulated) --
    cloned into all N_PARTICLES particles (shared policy, matches how
    fly_swarm_social.py trained once and cloned last night). None
    reproduces the original untrained/random-readout swarm."""
    flies = [CXModulatedBrain(sub, encode_idx, decode_idx, cx_idx, cfg, seed=1000 + i) for i in range(N_PARTICLES)]
    if trained_state is not None:
        for fly in flies:
            fly.readout_dir.data.copy_(trained_state["readout_dir"])
            fly.readout_gain.data.copy_(trained_state["readout_gain"])
            fly.readout_gain_bias.data.copy_(trained_state["readout_gain_bias"])
            fly.gain.data.copy_(trained_state["gain"])
            fly.bias.data.copy_(trained_state["bias"])
            fly.leak_logit.data.copy_(trained_state["leak_logit"])
    return flies


def run_swarm(sub, encode_idx, decode_idx, cx_idx, cfg, seed, trained_state=None):
    lo, hi = RASTRIGIN_BOUNDS
    rng = np.random.default_rng(seed)
    flies = build_swarm_brains(sub, encode_idx, decode_idx, cx_idx, cfg, trained_state=trained_state)
    for fly in flies:
        fly.h_state = torch.zeros(fly.n, 1)  # persistent state per particle, reset once per swarm run

    x = rng.uniform(lo, hi, size=(N_PARTICLES, DIM))
    v = rng.uniform(-1, 1, size=(N_PARTICLES, DIM))
    fx = np.array([rastrigin(xi) for xi in x])
    pbest_x, pbest_fx = x.copy(), fx.copy()
    gbest_i = int(np.argmin(pbest_fx))
    gbest_fx = float(pbest_fx[gbest_i])

    for it in range(N_ITERS):
        r1, r2 = rng.random((N_PARTICLES, DIM)), rng.random((N_PARTICLES, DIM))
        v = W * v + C1 * r1 * (pbest_x - x) + C2 * r2 * (pbest_x[gbest_i] - x)

        for i in range(N_PARTICLES):
            brain = flies[i]
            with torch.no_grad():
                cur_fx = fx[i]
                rays_np = _rays_for(x[i], cur_fx)
                w_eff = brain.effective_weights()
                a = torch.sigmoid(brain.leak_logit).unsqueeze(1)
                bias = brain.bias.unsqueeze(1)
                inject = torch.zeros(brain.n, 1)
                inject.index_add_(0, brain.encode_idx, torch.as_tensor(rays_np, dtype=torch.float32).unsqueeze(1))
                h = brain.h_state
                for _ in range(cfg.T_inner):
                    h = brain._substep(h, w_eff, a, bias, inject)
                brain.h_state = h
                decode_h = h.index_select(0, brain.decode_idx).squeeze(1)
                pred_dir = (brain.readout_dir @ decode_h).numpy()
                cx_h = h.index_select(0, brain.cx_idx).squeeze(1)
                gain = float(torch.sigmoid(brain.readout_gain @ cx_h + brain.readout_gain_bias).item())

            x[i] = np.clip(x[i] + v[i] + gain * pred_dir, lo, hi)

        fx = np.array([rastrigin(xi) for xi in x])
        improved = fx < pbest_fx
        pbest_x[improved] = x[improved]
        pbest_fx[improved] = fx[improved]
        gbest_i = int(np.argmin(pbest_fx))
        gbest_fx = min(gbest_fx, float(pbest_fx[gbest_i]))

    return gbest_fx


def _rays_for(x, fx):
    from flyopt.variants.fly_proposer_scene import _rays
    return _rays(x, rastrigin, fx, 0.3)


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    cx_pool = np.load(f"{CELLTYPES}/central_complex_indices.npy")

    encode_full, cx_verified = select_connected_encode_decode(
        base_weights, afferent, cx_pool, n_encode=N_RAYS, n_decode=N_DECODE + N_CX,
        max_hops=6, n_encode_candidates=200, seed=17000,
    )
    print(f"verified: {len(encode_full)} encode, {len(cx_verified)} CX")
    decode_full = cx_verified[:N_DECODE]
    cx_full = cx_verified[N_DECODE:N_DECODE + N_CX]

    all_targets = np.concatenate([decode_full, cx_full])
    sub_real, encode_idx, mapped_targets, _nodes = build_subgraph_bfs(base_weights, encode_full, all_targets, SUBGRAPH_SIZE, seed=17000)
    decode_idx = mapped_targets[: len(decode_full)]
    cx_idx = mapped_targets[len(decode_full):]
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = CXModulatedConfig(dim=DIM, T_inner=4, episode_steps=N_ITERS, ray_radius=0.3)
    # 2026-09-18: TRAINING uses a separate, shorter episode_steps=20 config
    # (same as the validated single-agent fly_cx_modulated_multiseed.py) --
    # NOT cfg's episode_steps=60. Training with episode_steps=60 backprops
    # through 60*T_inner=240 sequential recurrent steps in one episode,
    # which overflows to NaN gradients within 1-2 epochs at full 2000-node
    # scale (confirmed via direct diagnostic; grad-clipping alone does NOT
    # fix this since the NaN originates inside the backward pass itself,
    # before there is a finite gradient tensor to clip). The swarm's own
    # 60-iteration run never backprops (torch.no_grad()), so training on a
    # shorter horizon and deploying on a longer one is safe and matches
    # standard practice for recurrent search policies.
    train_cfg = CXModulatedConfig(dim=DIM, T_inner=4, episode_steps=20, ray_radius=0.3)
    log_path = "C:/projeler/fly_op/results/fly_cx_pso_swarm_trained_multiseed.jsonl"
    results = {"real_connectome": [], "er_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = er_null(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                # train ONE representative brain on this (kind, seed)'s substrate,
                # then clone its weights into all N_PARTICLES swarm members
                template = CXModulatedBrain(sub, encode_idx, decode_idx, cx_idx, train_cfg, seed=seed)
                train_cx_modulated(template, epochs=TRAIN_EPOCHS, lr=5e-3, seed=seed, geometries_per_step=6)
                trained_state = {
                    "readout_dir": template.readout_dir.data.clone(),
                    "readout_gain": template.readout_gain.data.clone(),
                    "readout_gain_bias": template.readout_gain_bias.data.clone(),
                    "gain": template.gain.data.clone(),
                    "bias": template.bias.data.clone(),
                    "leak_logit": template.leak_logit.data.clone(),
                }
                gbest_fx = run_swarm(sub, encode_idx, decode_idx, cx_idx, cfg, seed, trained_state=trained_state)
                results[name].append(gbest_fx)
                f.write(json.dumps({"seed": seed, "substrate": name, "gbest_fx": gbest_fx}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] gbest_fx={gbest_fx:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["er_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (CX-modulated PSO swarm, n=8 tohum) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_cx_pso_swarm_trained_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_cx_pso_swarm_trained_summary.json")


if __name__ == "__main__":
    main()
