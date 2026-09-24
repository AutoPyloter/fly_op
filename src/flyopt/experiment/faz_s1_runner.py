"""Faz S1 ("sahne") experiment orchestration (preregistration/faz_s1.md).

Run: python -m flyopt.experiment.faz_s1_runner [--smoke]

Same (kind, seed) parallelization and worker/thread settings as
faz1_runner.py (measured sweet spot on this machine, see that file's
THREADS_PER_WORKER comment) — per-iteration substrate cost (T LIF steps on
the full 139k-neuron graph) is unchanged from Faz 1/1b/1c, only what
surrounds it differs, so the same budget is expected to cost about the same
wall time.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import sparse

from flyopt.benchmarks import RASTRIGIN_BOUNDS, SPHERE_BOUNDS, rastrigin, sphere
from flyopt.experiment.faz1_runner import AFFERENT_POOL_PATH, EFFERENT_POOL_PATH, build_weights
from flyopt.experiment.runner import ExperimentRunner
from flyopt.experiment.search_loop import run_hillclimb
from flyopt.experiment.stats import holm_bonferroni, passes_falsification_gate, wilcoxon_paired
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from flyopt.variants.fly_proposer_scene import FlyProposerScene, SceneProposerConfig, calibrate_readout

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_RAW = REPO_ROOT / "results" / "raw"

SUBSTRATE_KINDS = ["flywire", "er_null", "degree_preserving_null", "weight_shuffle_null"]
MAIN_SEEDS = list(range(30))
TUNING_SEEDS = [1001, 1002, 1003]
GRID_RAY_SCALE = [0.5, 1.0, 2.0]  # same values as Faz 1's step_scale grid, for continuity
GRID_DECODE_SCALE = [0.2, 0.5, 1.0]
T_STEPS = 15
DIM = 10
N_READOUT = 50
RAY_RADIUS = 0.3  # ~6% of Rastrigin's +-5.12 range: local probe, not fixed
CALIB_STEPS = 100  # fixed, equal across substrates and seeds; not swept (scope control)
PROPOSER_SEED = 777  # fixed: encode/decode indices + initial readout are seed-only (PROTOCOL.md 4.1)
THREADS_PER_WORKER = 4


def _make_proposer(kind, seed, weights, ray_scale, decode_scale) -> tuple[FlyProposerScene, object]:
    encode_pool, decode_pool = np.load(AFFERENT_POOL_PATH), np.load(EFFERENT_POOL_PATH)
    substrate = SparseRecurrentSubstrate(weights)
    config = SceneProposerConfig(
        dim=DIM, n_readout=N_READOUT, T=T_STEPS, ray_radius=RAY_RADIUS,
        ray_scale=ray_scale, decode_scale=decode_scale,
    )
    proposer = FlyProposerScene(
        substrate, config, seed=PROPOSER_SEED,
        encode_pool=encode_pool, decode_pool=decode_pool, reset_seed=seed,
    )
    return proposer, substrate


def _tune_worker(kind: str, seed: int, grid, budget: int) -> dict:
    import torch

    torch.set_num_threads(THREADS_PER_WORKER)
    base_weights = sparse.load_npz(DATA_PROCESSED / "adjacency.npz")
    weights = build_weights(kind, seed, base_weights)
    out = {}
    for ray_scale, decode_scale in grid:
        proposer, _ = _make_proposer(kind, seed, weights, ray_scale, decode_scale)
        proposer.bind_objective(rastrigin)
        calibrate_readout(proposer, rastrigin, dim=DIM, bounds=RASTRIGIN_BOUNDS, n_steps=CALIB_STEPS, seed=seed)
        best_fx = run_hillclimb(proposer, rastrigin, dim=DIM, bounds=RASTRIGIN_BOUNDS, budget=budget, seed=seed)
        out[f"{ray_scale}|{decode_scale}"] = best_fx
    return {"kind": kind, "seed": seed, "results": out}


def _main_worker(kind: str, seed: int, ray_scale: float, decode_scale: float, main_budget: int, sphere_budget: int) -> dict:
    import torch

    torch.set_num_threads(THREADS_PER_WORKER)
    t0 = time.time()
    base_weights = sparse.load_npz(DATA_PROCESSED / "adjacency.npz")
    weights = build_weights(kind, seed, base_weights)
    null_gen_time = time.time() - t0

    t0 = time.time()
    proposer, _ = _make_proposer(kind, seed, weights, ray_scale, decode_scale)
    proposer.bind_objective(rastrigin)
    calibrate_readout(proposer, rastrigin, dim=DIM, bounds=RASTRIGIN_BOUNDS, n_steps=CALIB_STEPS, seed=seed)
    calib_time = time.time() - t0

    t0 = time.time()
    rastrigin_best = run_hillclimb(proposer, rastrigin, dim=DIM, bounds=RASTRIGIN_BOUNDS, budget=main_budget, seed=seed)
    rastrigin_time = time.time() - t0

    t0 = time.time()
    proposer2, _ = _make_proposer(kind, seed, weights, ray_scale, decode_scale)
    proposer2.bind_objective(sphere)
    calibrate_readout(proposer2, sphere, dim=DIM, bounds=SPHERE_BOUNDS, n_steps=CALIB_STEPS, seed=seed)
    sphere_best = run_hillclimb(proposer2, sphere, dim=DIM, bounds=SPHERE_BOUNDS, budget=sphere_budget, seed=seed)
    sphere_time = time.time() - t0

    return {
        "kind": kind, "seed": seed, "ray_scale": ray_scale, "decode_scale": decode_scale,
        "rastrigin_best_fx": rastrigin_best, "sphere_best_fx": sphere_best,
        "null_gen_time_s": null_gen_time, "calib_time_s": calib_time,
        "rastrigin_time_s": rastrigin_time, "sphere_time_s": sphere_time,
    }


def run_faz_s1(
    main_budget: int = 1500,
    sphere_budget: int = 200,
    tuning_budget: int = 100,
    seeds: list[int] = None,
    max_workers: int = 2,
    log_path: Path = None,
) -> dict:
    seeds = seeds if seeds is not None else MAIN_SEEDS
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    log_path = log_path or (RESULTS_RAW / "faz_s1_progress.jsonl")
    grid = list(itertools.product(GRID_RAY_SCALE, GRID_DECODE_SCALE))

    def log(entry: dict):
        entry["ts"] = time.time()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    log({"event": "start", "seeds": seeds, "main_budget": main_budget, "tuning_budget": tuning_budget, "calib_steps": CALIB_STEPS})

    best_hparams: dict[str, tuple[float, float]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_tune_worker, kind, seed, grid, tuning_budget): (kind, seed)
            for kind in SUBSTRATE_KINDS for seed in TUNING_SEEDS
        }
        agg: dict[str, dict[str, list[float]]] = {
            k: {f"{r}|{d}": [] for r, d in grid} for k in SUBSTRATE_KINDS
        }
        for fut in as_completed(futures):
            res = fut.result()
            for key, val in res["results"].items():
                agg[res["kind"]][key].append(val)
            log({"event": "tune_seed_done", "kind": res["kind"], "seed": res["seed"]})

    for kind in SUBSTRATE_KINDS:
        means = {key: float(np.mean(vals)) for key, vals in agg[kind].items()}
        best_key = min(means, key=means.get)
        r, d = best_key.split("|")
        best_hparams[kind] = (float(r), float(d))
        log({"event": "tune_done", "kind": kind, "best_ray_scale": float(r), "best_decode_scale": float(d), "mean_best_fx": means[best_key]})

    raw_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for kind in SUBSTRATE_KINDS:
            ray_scale, decode_scale = best_hparams[kind]
            for seed in seeds:
                fut = ex.submit(_main_worker, kind, seed, ray_scale, decode_scale, main_budget, sphere_budget)
                futures[fut] = (kind, seed)
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            raw_results.append(res)
            done += 1
            log({"event": "main_run_done", "progress": f"{done}/{len(futures)}", **res})

    return {"best_hparams": best_hparams, "raw_results": raw_results, "seeds": seeds, "main_budget": main_budget}


def analyze(result: dict) -> dict:
    by_kind = {k: {} for k in SUBSTRATE_KINDS}
    for r in result["raw_results"]:
        by_kind[r["kind"]][r["seed"]] = r["rastrigin_best_fx"]

    seeds = result["seeds"]
    flywire = np.array([by_kind["flywire"][s] for s in seeds])
    degree_null = np.array([by_kind["degree_preserving_null"][s] for s in seeds])
    er = np.array([by_kind["er_null"][s] for s in seeds])
    weight_shuffle_arr = np.array([by_kind["weight_shuffle_null"][s] for s in seeds])

    gate_result = wilcoxon_paired(flywire, degree_null)
    gate_pass = passes_falsification_gate(gate_result)

    secondary = {
        "flywire_vs_er_null": wilcoxon_paired(flywire, er),
        "flywire_vs_weight_shuffle_null": wilcoxon_paired(flywire, weight_shuffle_arr),
    }
    p_values = [secondary[k].p_value for k in secondary]
    holm_reject = holm_bonferroni(p_values)

    return {
        "gate": {
            "test": gate_result.test, "p_value": gate_result.p_value, "effect_size": gate_result.effect_size,
            "passes_falsification_gate": gate_pass,
            "flywire_mean": float(flywire.mean()), "degree_preserving_null_mean": float(degree_null.mean()),
        },
        "secondary": {
            k: {"p_value": v.p_value, "effect_size": v.effect_size, "holm_reject": holm_reject[i]}
            for i, (k, v) in enumerate(secondary.items())
        },
        "medians": {
            "flywire": float(np.median(flywire)), "degree_preserving_null": float(np.median(degree_null)),
            "er_null": float(np.median(er)), "weight_shuffle_null": float(np.median(weight_shuffle_arr)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny budgets, 2 seeds, for pipeline validation")
    parser.add_argument(
        "--pilot-seeds", type=int, default=None,
        help="run only the first N of MAIN_SEEDS (full pre-registered main_budget/sphere_budget unchanged) "
        "-- a resource-driven scope reduction decided before seeing any main-run result, same precedent as "
        "Faz 1's original 10000->1500 budget cut (EXPERIMENTS.md 2026-09-14); logged here, not in "
        "preregistration/faz_s1.md, which stays immutable.",
    )
    args = parser.parse_args()

    if not (AFFERENT_POOL_PATH.exists() and EFFERENT_POOL_PATH.exists()):
        raise SystemExit(f"Faz S1 requires {AFFERENT_POOL_PATH} and {EFFERENT_POOL_PATH} (already built for Faz 1b/1c).")

    log_path = RESULTS_RAW / "faz_s1_progress.jsonl"
    if args.smoke:
        result = run_faz_s1(main_budget=50, sphere_budget=20, tuning_budget=10, seeds=[0, 1], max_workers=4, log_path=log_path)
    elif args.pilot_seeds:
        result = run_faz_s1(seeds=MAIN_SEEDS[: args.pilot_seeds], log_path=log_path)
    else:
        result = run_faz_s1(seeds=MAIN_SEEDS, log_path=log_path)

    analysis = analyze(result)
    print(json.dumps(analysis, indent=2))

    runner = ExperimentRunner(REPO_ROOT, REPO_ROOT / "results")
    record = runner.start(
        config={"phase": "faz_s1", "smoke": args.smoke, "best_hparams": result["best_hparams"], "main_budget": result["main_budget"]},
        seed=0,
    )
    positive = analysis["gate"]["passes_falsification_gate"]
    out_path = runner.finish(record, {"analysis": analysis, "raw_results": result["raw_results"]}, positive=positive)
    print(f"\nWrote {out_path}")
    print(f"\nGATE {'PASSED' if positive else 'FAILED'} (degree_preserving_null vs flywire, PROTOCOL.md section 2, phase=faz_s1)")


if __name__ == "__main__":
    main()
