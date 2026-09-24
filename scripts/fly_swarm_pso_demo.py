"""Fly-Hybrid PSO demo + development ideas (PROTOCOL.md 2.7). NOT a
pre-registered phase, no falsification gate, no null-model comparison --
a visual/curiosity exploration requested 2026-09-17, developed further per
2026-09-17 follow-up ("PSO'yu gelistirmek icin fikirlerini bekliyorum").

Four variants, same initial swarm state (seed=42) so differences are
attributable to the proposer mechanism, not luck:

  classic       -- plain PSO, no connectome at all (the control the user
                   asked for: does the fly nudge actually help?)
  raw           -- original demo: FlyProposer, raw-x injection, constant
                   fly_weight (what we already watched and liked)
  scene         -- FlyProposerScene (Faz S1's ray/gradient encoding +
                   imitation-calibrated readout) instead of raw-x injection,
                   constant fly_weight
  scene_anneal  -- scene, but fly_weight decays from FLY_WEIGHT_START to
                   FLY_WEIGHT_END over the run (explore early, exploit late)

Writes a 4-line convergence plot (gbest_fx per iteration) and a GIF of the
scene_anneal swarm.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from flyopt.benchmarks import RASTRIGIN_BOUNDS, rastrigin
from flyopt.substrate import SearchContext
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from flyopt.variants.fly_proposer import FlyProposer, FlyProposerConfig
from flyopt.variants.fly_proposer_scene import FlyProposerScene, SceneProposerConfig, calibrate_readout

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DIM = 2
N_PARTICLES = 12
N_ITERS = 60
FLY_WEIGHT = 0.6  # constant weight for "raw" and "scene"
FLY_WEIGHT_START, FLY_WEIGHT_END = 1.0, 0.05  # for "scene_anneal"
W, C1, C2 = 0.6, 1.2, 1.2  # PSO inertia, cognitive, social
AFFERENT_POOL = f"{DATA_PROCESSED}/afferent_indices.npy"
EFFERENT_POOL = f"{DATA_PROCESSED}/efferent_indices.npy"


def rastrigin2d_grid(bounds, n=200):
    lo, hi = bounds
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, ys)
    Z = 20 + X**2 - 10 * np.cos(2 * np.pi * X) + Y**2 - 10 * np.cos(2 * np.pi * Y)
    return X, Y, Z


def make_flies(variant: str, substrate):
    if variant == "classic":
        return None
    if variant == "raw":
        return [
            FlyProposer(substrate, FlyProposerConfig(dim=DIM, n_readout=50, T=15), seed=1000 + i)
            for i in range(N_PARTICLES)
        ]
    # scene / scene_anneal
    encode_pool, decode_pool = np.load(AFFERENT_POOL), np.load(EFFERENT_POOL)
    flies = []
    for i in range(N_PARTICLES):
        cfg = SceneProposerConfig(dim=DIM, n_readout=50, T=15, ray_radius=0.3, ray_scale=1.0, decode_scale=0.5)
        fly = FlyProposerScene(substrate, cfg, seed=1000 + i, encode_pool=encode_pool, decode_pool=decode_pool, reset_seed=i)
        fly.bind_objective(rastrigin)
        calibrate_readout(fly, rastrigin, dim=DIM, bounds=RASTRIGIN_BOUNDS, n_steps=100, seed=1000 + i)
        flies.append(fly)
    return flies


def fly_weight_fn(variant: str, it: int) -> float:
    if variant == "classic":
        return 0.0
    if variant != "scene_anneal":
        return FLY_WEIGHT
    frac = it / max(1, N_ITERS - 1)
    return FLY_WEIGHT_START * (1 - frac) + FLY_WEIGHT_END * frac


def reset_flies(flies, trial_seed: int) -> None:
    """Clear persistent substrate state before a fresh trial (FlyProposerScene
    keeps state across the whole run by design, see fly_proposer_scene.py) so
    multiple seed trials don't leak state from one into the next. Calibration
    (readout_W) is untouched -- it's a property of the fly, not the trial."""
    if not flies:
        return
    for fly in flies:
        if hasattr(fly, "_state"):
            fly._state = None
            fly._reset_seed = trial_seed


def run_variant(variant: str, substrate, seed: int = 42, flies=None):
    if flies is None:
        flies = make_flies(variant, substrate)
    else:
        reset_flies(flies, trial_seed=seed)

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

        fw = fly_weight_fn(variant, it)
        for i in range(N_PARTICLES):
            step = v[i]
            if flies is not None and fw > 0:
                ctx = SearchContext(iteration=it, budget_total=N_ITERS, budget_used=it, best_x=pbest_x[i], best_fx=pbest_fx[i])
                fly_delta = flies[i].propose(x[i], ctx) - x[i]
                step = step + fw * fly_delta
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

    print(f"[{variant:14s}] final gbest_fx={gbest_fx:8.4f}  gbest_x={gbest_x}")
    return history, curve, gbest_x


def render_gif(history, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    X, Y, Z = rastrigin2d_grid(RASTRIGIN_BOUNDS)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, Z, levels=40, cmap="viridis")
    ax.plot(0, 0, "w*", markersize=18, label="gercek minimum (0,0)")
    scat = ax.scatter([], [], c="red", s=60, edgecolors="white", label="sinekler")
    ax.legend(loc="upper right")
    ax.set_title(title)

    writer = PillowWriter(fps=8)
    with writer.saving(fig, out_path, dpi=100):
        for frame in history:
            scat.set_offsets(frame)
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out_path}")


def render_convergence(curves: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, curve in curves.items():
        ax.plot(curve, label=name)
    ax.set_xlabel("iterasyon")
    ax.set_ylabel("gbest_fx (Rastrigin, dusuk=iyi)")
    ax.set_title("Fly-Hybrid PSO varyantlari: yakinsama karsilastirmasi")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    substrate = SparseRecurrentSubstrate(weights)  # built once, shared read-only across all variants/particles

    curves = {}
    histories = {}
    for variant in ["classic", "raw", "scene", "scene_anneal"]:
        history, curve, gbest_x = run_variant(variant, substrate)
        curves[variant] = curve
        histories[variant] = history

    render_convergence(curves, "C:/projeler/fly_op/results/fly_swarm_pso_convergence.png")
    render_gif(
        histories["scene_anneal"],
        "C:/projeler/fly_op/results/fly_swarm_pso_demo_scene_anneal.gif",
        "Fly-Hybrid PSO (isin kodlama + azalan agirlik), 2-D Rastrigin",
    )


if __name__ == "__main__":
    main()
