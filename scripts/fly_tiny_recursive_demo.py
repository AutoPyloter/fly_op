"""TRM (Tiny Recursive Model) -style proposer vs. classic PSO vs. the
connectome variants (informal, 2026-09-17 -- "sinek beyni yerine HRM/TRM
mimarisi denesek nasil sonuc verir?"). Not pre-registered.

No connectome/LIF simulation is used here at all -- the whole point is to
swap the substrate for a small, fully-trainable-by-backprop recursive
network and see whether THIS kind of small network can do any better than
classic PSO on the exact same task/loop/seeds that every connectome
variant (raw/scene/scene_anneal/social/social_anneal, see
fly_swarm_pso_demo.py and fly_swarm_social_demo.py) failed to beat.
"""
from __future__ import annotations

import json

import numpy as np

from flyopt.benchmarks import RASTRIGIN_BOUNDS, rastrigin
from flyopt.substrate import SearchContext
from flyopt.variants.tiny_recursive_proposer import TinyRecursiveConfig, build_and_train

import fly_swarm_pso_demo as base

DIM = base.DIM
N_PARTICLES = base.N_PARTICLES
N_ITERS = base.N_ITERS
W, C1, C2 = base.W, base.C1, base.C2
FLY_WEIGHT = base.FLY_WEIGHT
FLY_WEIGHT_START, FLY_WEIGHT_END = base.FLY_WEIGHT_START, base.FLY_WEIGHT_END
SEEDS = list(range(8))  # same 8 seeds as the connectome multiseed runs


def run_variant(variant: str, proposer, seed: int):
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

        if variant == "classic":
            fw = 0.0
        elif variant == "tiny_recursive_anneal":
            frac = it / max(1, N_ITERS - 1)
            fw = FLY_WEIGHT_START * (1 - frac) + FLY_WEIGHT_END * frac
        else:
            fw = FLY_WEIGHT

        for i in range(N_PARTICLES):
            step = v[i]
            if proposer is not None and fw > 0:
                ctx = SearchContext(iteration=it, budget_total=N_ITERS, budget_used=it, best_x=pbest_x[i], best_fx=pbest_fx[i])
                delta = proposer.propose(x[i], ctx) - x[i]
                step = step + fw * delta
            x[i] = np.clip(x[i] + step, lo, hi)

        fx = np.array([rastrigin(xi) for xi in x])
        improved = fx < pbest_fx
        pbest_x[improved] = x[improved]
        pbest_fx[improved] = fx[improved]
        gbest_i = int(np.argmin(pbest_fx))
        if pbest_fx[gbest_i] < gbest_fx:
            gbest_x, gbest_fx = pbest_x[gbest_i].copy(), float(pbest_fx[gbest_i])

        history.append(x.copy())
        curve.append(gbest_fx)

    return history, curve, gbest_x


def main():
    cfg = TinyRecursiveConfig(dim=DIM, latent_dim=32, hidden_dim=64, n_recursions=6, ray_radius=0.3, decode_scale=0.5)
    proposer = build_and_train(cfg, RASTRIGIN_BOUNDS, n_problems=25, n_starts=12, epochs=300, lr=1e-3, seed=3000)
    proposer.bind_objective(rastrigin)  # deployment: the real, non-offset landscape

    variants = ["classic", "tiny_recursive", "tiny_recursive_anneal"]
    results: dict[str, list[float]] = {v: [] for v in variants}
    histories_seed0: dict[str, list] = {}

    log_path = "C:/projeler/fly_op/results/fly_tiny_recursive_multiseed.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            for variant in variants:
                p = None if variant == "classic" else proposer
                history, curve, gbest_x = run_variant(variant, p, seed=seed)
                fx = float(np.sum(np.asarray(gbest_x) ** 2 - 10 * np.cos(2 * np.pi * np.asarray(gbest_x))) + 20)
                results[variant].append(fx)
                if seed == 0:
                    histories_seed0[variant] = history
                f.write(json.dumps({"seed": seed, "variant": variant, "gbest_fx": fx, "gbest_x": list(map(float, gbest_x))}) + "\n")
                f.flush()
                print(f"seed={seed} [{variant:22s}] gbest_fx={fx:8.4f}")

    print("\n=== ozet (n=%d tohum) ===" % len(SEEDS))
    for variant in variants:
        vals = np.array(results[variant])
        n_solved = int(np.sum(vals < 0.1))
        print(f"{variant:22s} mean={vals.mean():8.4f} median={np.median(vals):8.4f} "
              f"min={vals.min():8.4f} max={vals.max():8.4f}  gercek_minimumu_buldu={n_solved}/{len(SEEDS)}")

    curves_seed0 = {}
    for variant in variants:
        best = np.inf
        curve = []
        for frame in histories_seed0[variant]:
            fxs = np.array([rastrigin(p) for p in frame])
            best = min(best, float(fxs.min()))
            curve.append(best)
        curves_seed0[variant] = curve
    base.render_convergence(curves_seed0, "C:/projeler/fly_op/results/fly_tiny_recursive_convergence_seed0.png")
    base.render_gif(
        histories_seed0["tiny_recursive_anneal"],
        "C:/projeler/fly_op/results/fly_tiny_recursive_anneal_demo.gif",
        "TRM-tarzi kucuk ozyinelemeli ag (connectome yok), 2-D Rastrigin",
    )


if __name__ == "__main__":
    main()
