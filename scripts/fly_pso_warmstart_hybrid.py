"""Fly+PSO warm-start hybrid (2026-09-23, user's own idea: "PSO da rasgele
noktadan başlamak yerine sinekler ile önceden başlangıç noktası verilse
nasıl olur?"). Uses the already-validated single-shot RateBrain (official
finding: real vs degree_preserving_rewire, delta=0.884, n=30) to seed
PSO's initial population instead of pure uniform-random init, on the SAME
slope-stability factor-of-safety search task.

Critically, this is NOT "fly vs PSO" (that comparison axis was already
shown to be uninformative about connectome structure -- see
sensory_fly_nullcontrol_benchmark.py and EXPERIMENTS.md 2026-09-22). This
tests whether the REAL connectome's warm-start suggestions help PSO more
than the NULL (degree_preserving_rewire) connectome's warm-start
suggestions do -- isolating the connectome's structural contribution to
warm-starting, not just "does any trained network's guess help a
little" (a null-connectome-informed guess is still a trained,
non-trivial guess, so it controls for that).

Three arms per test geometry:
  a) baseline: PSO with pure uniform-random initial population
  b) real-warmstart: PSO initial population seeded by the trained REAL
     connectome's single-shot direction predictions
  c) null-warmstart: same, but using the trained NULL (degree_preserving)
     connectome

Official comparison: (b) vs (c), same stats convention as everywhere
else (Wilcoxon, Mann-Whitney, Cliff's delta) -- LOWER final FS is better.
"""
from __future__ import annotations

import gc
import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry, sample_valid_point
from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import RateBrain, RateBrainConfig, build_subgraph_bfs, select_connected_encode_decode, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0

NUM_PARTICLES = 15   # 2026-09-23: reduced from 300 -- smoke-tested budgets showed 300x80 gives PSO
PSO_ITERS = 10       # so much power that random/real/null warm-starts all converge to the identical
                     # optimum (std=0.0001 across 10 random-init runs) -- warm-starting can only matter
                     # if PSO itself is resource-constrained. 15x10 gives real random-init variance
                     # (std=0.023 across 10 runs, measured), a budget where a good starting point
                     # plausibly helps.
N_WARMSTART_SEEDS = 8    # scaled down with the population (was 40 for a 300-particle population)
JITTER_FRAC = 0.05       # small jitter around each warm-start point to fill out the population diversity
SEEDS = list(range(8))   # n=8 pilot: one (geometry-set, real-brain-train-seed, null-rewiring-seed) per rep


def build_training_set_geo(n_problems, n_starts, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = random_geometry(rng)
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = sample_valid_point(geo, rng)
            fx = f(x)
            rays = _rays(x, f, fx, RAY_RADIUS)
            target = _teacher_delta(x, rays, RAY_RADIUS)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(DIM)
            X.append(rays); Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def pso_search(geo, init_population: np.ndarray, iters: int, rng: np.random.Generator) -> float:
    """Standard gbest-PSO in numpy (small population/dim, CPU is fine and
    keeps this decoupled from the brain's own GPU tensors). Returns best
    (lowest) factor of safety found. init_population: (num_particles, DIM)."""
    lo, hi = geo.param_bounds()
    x = np.clip(init_population.copy(), lo, hi)
    v = np.zeros_like(x)
    obj = np.array([factor_of_safety(xi, geo) for xi in x])
    pbest, pbest_obj = x.copy(), obj.copy()
    gbest = pbest[np.argmin(pbest_obj)].copy()
    gbest_obj = pbest_obj.min()
    w, c1, c2 = 0.5, 1.5, 1.5
    for _ in range(iters):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = np.clip(x + v, lo, hi)
        obj = np.array([factor_of_safety(xi, geo) for xi in x])
        mask = obj < pbest_obj
        pbest[mask], pbest_obj[mask] = x[mask], obj[mask]
        if pbest_obj.min() < gbest_obj:
            gbest_obj = pbest_obj.min()
            gbest = pbest[np.argmin(pbest_obj)].copy()
    return float(gbest_obj)


def fly_warmstart_population(brain, geo, rng: np.random.Generator) -> np.ndarray:
    """N_WARMSTART_SEEDS random valid starts, each advanced one single-shot
    fly-predicted step; population filled out to NUM_PARTICLES by jittering
    around these proposed points (keeps the same "informed neighborhood"
    spirit as a warm start, rather than collapsing to N_WARMSTART_SEEDS
    duplicate points)."""
    lo, hi = geo.param_bounds()
    span = hi - lo
    f = lambda p: factor_of_safety(p, geo)
    starts = []
    for _ in range(N_WARMSTART_SEEDS):
        x0 = sample_valid_point(geo, rng)
        fx0 = f(x0)
        rays = _rays(x0, f, fx0, RAY_RADIUS)
        with torch.no_grad():
            pred = brain(torch.as_tensor(rays, device=DEVICE, dtype=torch.float32).unsqueeze(0))
        step = pred.squeeze(0).cpu().numpy()
        x1 = np.clip(x0 + step, lo, hi)
        starts.append(x1)
    starts = np.array(starts)

    pop = []
    for _ in range(NUM_PARTICLES):
        base = starts[rng.integers(len(starts))]
        pop.append(base + rng.normal(0, JITTER_FRAC, size=DIM) * span)
    return np.clip(np.array(pop), lo, hi)


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, {len(encode_idx)} encode / {len(decode_idx)} decode")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_pso_warmstart_multiseed.jsonl"
    results = {"baseline": [], "real_warmstart": [], "null_warmstart": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            brain_real = RateBrain(sub_real, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
            train(brain_real, X, Y, epochs=EPOCHS, lr=LR)
            brain_null = RateBrain(sub_null, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
            train(brain_null, X, Y, epochs=EPOCHS, lr=LR)

            rng = np.random.default_rng(20000 + seed)
            geo = random_geometry(rng)
            lo, hi = geo.param_bounds()

            baseline_pop = rng.uniform(lo, hi, size=(NUM_PARTICLES, DIM))
            real_pop = fly_warmstart_population(brain_real, geo, np.random.default_rng(30000 + seed))
            null_pop = fly_warmstart_population(brain_null, geo, np.random.default_rng(30000 + seed))

            pso_rng = np.random.default_rng(40000 + seed)
            baseline_fs = pso_search(geo, baseline_pop, PSO_ITERS, np.random.default_rng(40000 + seed))
            real_fs = pso_search(geo, real_pop, PSO_ITERS, np.random.default_rng(40000 + seed))
            null_fs = pso_search(geo, null_pop, PSO_ITERS, np.random.default_rng(40000 + seed))

            results["baseline"].append(baseline_fs)
            results["real_warmstart"].append(real_fs)
            results["null_warmstart"].append(null_fs)
            f.write(json.dumps({"seed": seed, "baseline": baseline_fs, "real_warmstart": real_fs, "null_warmstart": null_fs}) + "\n")
            f.flush()
            print(f"seed={seed}: baseline={baseline_fs:.4f}  real_warmstart={real_fs:.4f}  null_warmstart={null_fs:.4f}")
            del brain_real, brain_null
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    real = np.array(results["real_warmstart"])
    null = np.array(results["null_warmstart"])
    baseline = np.array(results["baseline"])
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])  # lower FS = better search
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33

    print("\n=== SONUC (Fly+PSO warm-start, n=8) ===")
    print(f"baseline (random init) medyan FS:       {np.median(baseline):.4f}")
    print(f"real-connectome warmstart medyan FS:    {np.median(real):.4f}")
    print(f"null-connectome warmstart medyan FS:    {np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta (real vs null)={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_pso_warmstart_summary.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline.tolist(), "real_warmstart": real.tolist(), "null_warmstart": null.tolist(),
                   "wilcoxon_p": float(w_p), "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta,
                   "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_pso_warmstart_summary.json")


if __name__ == "__main__":
    main()
