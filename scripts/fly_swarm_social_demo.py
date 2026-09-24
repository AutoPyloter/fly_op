"""Fly-Swarm-Social: "son bir deneme" (2026-09-17) -- richer multi-problem
training + angular social-awareness rays, tested against the same 6
variants/8-seed protocol as fly_swarm_pso_multiseed.py for a fair
comparison. Still informal, not pre-registered.

Adds two new variants to the earlier four (classic/raw/scene/scene_anneal,
see fly_swarm_pso_demo.py):

  social        -- FlySwarmSocial: ridge-regressed against a dataset built
                   from MANY randomly-offset Rastrigin instances x many
                   random starts (not one scripted rollout on one
                   landscape), input includes angular "social rays"
                   (bearing/distance/fitness-as-height of other flies),
                   target blends "toward the true minimum" with "toward
                   the best nearby neighbor". Constant fly_weight.
  social_anneal -- same proposer, annealed fly_weight (like scene_anneal).

Real (non-offset) 2-D Rastrigin is the actual test landscape; the offset
instances are training-only synthetic variety.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from flyopt.benchmarks import RASTRIGIN_BOUNDS, rastrigin
from flyopt.substrate import SearchContext
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from flyopt.variants.fly_swarm_social import (
    FlySwarmSocial,
    SwarmSocialConfig,
    build_training_set,
    calibrate_from_dataset,
)

import fly_swarm_pso_demo as base

DATA_PROCESSED = base.DATA_PROCESSED
DIM = base.DIM
N_PARTICLES = base.N_PARTICLES
N_ITERS = base.N_ITERS
W, C1, C2 = base.W, base.C1, base.C2
FLY_WEIGHT = base.FLY_WEIGHT
FLY_WEIGHT_START, FLY_WEIGHT_END = base.FLY_WEIGHT_START, base.FLY_WEIGHT_END
SEEDS = list(range(8))  # same 8 seeds as fly_swarm_pso_multiseed.py

N_TRAIN_PROBLEMS, N_TRAIN_STARTS, N_PHANTOM_NEIGHBORS = 25, 12, 4


def build_social_swarm(substrate):
    encode_pool, decode_pool = np.load(base.AFFERENT_POOL), np.load(base.EFFERENT_POOL)
    cfg = SwarmSocialConfig(dim=DIM, n_readout=50, T=15, ray_radius=0.3, ray_scale=1.0,
                             n_social_rays=8, social_scale=1.0, decode_scale=0.5,
                             social_teacher_weight=0.5)
    template_seed = 2000
    template = FlySwarmSocial(substrate, cfg, seed=template_seed, encode_pool=encode_pool,
                               decode_pool=decode_pool, reset_seed=0)
    X, Y = build_training_set(template, DIM, RASTRIGIN_BOUNDS, N_TRAIN_PROBLEMS,
                               N_TRAIN_STARTS, N_PHANTOM_NEIGHBORS, seed=template_seed)
    calibrate_from_dataset(template, X, Y)
    print(f"[social calib] {X.shape[0]} samples, readout_W norm={np.linalg.norm(template.readout_W):.3f}")

    flies = []
    for i in range(N_PARTICLES):
        fly = FlySwarmSocial(substrate, cfg, seed=template_seed, encode_pool=encode_pool,
                              decode_pool=decode_pool, reset_seed=i)
        fly.readout_W = template.readout_W.copy()
        fly.bind_objective(rastrigin)  # deployment: the real, non-offset landscape
        flies.append(fly)
    return flies


def run_social_variant(variant: str, substrate, flies, seed: int = 42):
    for fly in flies:
        fly._state = None
        fly._reset_seed = seed

    rng = np.random.default_rng(seed)
    lo, hi = RASTRIGIN_BOUNDS
    x = rng.uniform(lo, hi, size=(N_PARTICLES, DIM))
    v = rng.uniform(-1, 1, size=(N_PARTICLES, DIM))
    fx = np.array([rastrigin(xi) for xi in x])
    pbest_x, pbest_fx = x.copy(), fx.copy()
    gbest_i = int(np.argmin(pbest_fx))
    gbest_x, gbest_fx = pbest_x[gbest_i].copy(), float(pbest_fx[gbest_i])

    history = [x.copy()]
    curve = [gbest_fx]
    for it in range(N_ITERS):
        r1, r2 = rng.random((N_PARTICLES, DIM)), rng.random((N_PARTICLES, DIM))
        v = W * v + C1 * r1 * (pbest_x - x) + C2 * r2 * (gbest_x - x)

        fw = FLY_WEIGHT if variant == "social" else (
            FLY_WEIGHT_START * (1 - it / max(1, N_ITERS - 1)) + FLY_WEIGHT_END * (it / max(1, N_ITERS - 1))
        )
        for i in range(N_PARTICLES):
            others = [j for j in range(N_PARTICLES) if j != i]
            flies[i].set_neighbors(x[others], fx[others])
            ctx = SearchContext(iteration=it, budget_total=N_ITERS, budget_used=it, best_x=pbest_x[i], best_fx=pbest_fx[i])
            fly_delta = flies[i].propose(x[i], ctx) - x[i]
            x[i] = np.clip(x[i] + v[i] + fw * fly_delta, lo, hi)

        fx = np.array([rastrigin(xi) for xi in x])
        improved = fx < pbest_fx
        pbest_x[improved] = x[improved]
        pbest_fx[improved] = fx[improved]
        gbest_i = int(np.argmin(pbest_fx))
        if pbest_fx[gbest_i] < gbest_fx:
            gbest_x, gbest_fx = pbest_x[gbest_i].copy(), float(pbest_fx[gbest_i])

        history.append(x.copy())
        curve.append(gbest_fx)

    print(f"seed={seed} [{variant:14s}] gbest_fx={gbest_fx:8.4f}")
    return history, curve, gbest_x


def main():
    weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    substrate = SparseRecurrentSubstrate(weights)

    social_flies = build_social_swarm(substrate)

    old_flies = {
        "classic": None,
        "raw": base.make_flies("raw", substrate),
        "scene": base.make_flies("scene", substrate),
    }
    old_flies["scene_anneal"] = old_flies["scene"]

    variants = ["classic", "raw", "scene", "scene_anneal", "social", "social_anneal"]
    results: dict[str, list[float]] = {v: [] for v in variants}
    histories_seed0: dict[str, list] = {}

    log_path = "C:/projeler/fly_op/results/fly_swarm_social_multiseed.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            for variant in variants:
                if variant in ("social", "social_anneal"):
                    history, curve, gbest_x = run_social_variant(variant, substrate, social_flies, seed=seed)
                else:
                    history, curve, gbest_x = base.run_variant(variant, substrate, seed=seed, flies=old_flies[variant])
                fx = float(np.sum(np.asarray(gbest_x) ** 2 - 10 * np.cos(2 * np.pi * np.asarray(gbest_x))) + 20)
                results[variant].append(fx)
                if seed == 0:
                    histories_seed0[variant] = history
                f.write(json.dumps({"seed": seed, "variant": variant, "gbest_fx": fx, "gbest_x": list(map(float, gbest_x))}) + "\n")
                f.flush()
                print(f"seed={seed} [{variant:14s}] gbest_fx={fx:8.4f}")

    print("\n=== ozet (n=%d tohum) ===" % len(SEEDS))
    for variant in variants:
        vals = np.array(results[variant])
        n_solved = int(np.sum(vals < 0.1))
        print(f"{variant:14s} mean={vals.mean():8.4f} median={np.median(vals):8.4f} "
              f"min={vals.min():8.4f} max={vals.max():8.4f}  gercek_minimumu_buldu={n_solved}/{len(SEEDS)}")

    # convergence plot from seed-0 histories (best-so-far fx per iteration)
    curves_seed0 = {}
    for variant in variants:
        hx = histories_seed0[variant]
        curve = []
        best = np.inf
        for frame in hx:
            fxs = np.array([rastrigin(p) for p in frame])
            best = min(best, float(fxs.min()))
            curve.append(best)
        curves_seed0[variant] = curve
    base.render_convergence(curves_seed0, "C:/projeler/fly_op/results/fly_swarm_social_convergence_seed0.png")

    base.render_gif(
        histories_seed0["social_anneal"],
        "C:/projeler/fly_op/results/fly_swarm_social_anneal_demo.gif",
        "Fly-Swarm-Social (cok-problem egitim + sosyal isinlar), 2-D Rastrigin",
    )


if __name__ == "__main__":
    main()
