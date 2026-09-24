"""Faz 1 gate experiment orchestration (preregistration/faz_1.md).

Run: python -m flyopt.experiment.faz1_runner [--smoke]

Parallelized across (substrate_kind, seed) with ProcessPoolExecutor,
THREADS_PER_WORKER torch threads per worker (see that constant's comment —
single-threading each worker was tried first and measured to be much
slower in aggregate, not faster, because CSR sparse.mm itself scales well
with threads).
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
from flyopt.experiment.runner import ExperimentRunner
from flyopt.experiment.search_loop import run_hillclimb
from flyopt.experiment.stats import (
    holm_bonferroni,
    mann_whitney,
    passes_falsification_gate,
    wilcoxon_paired,
)
from flyopt.substrates.graph_builders import degree_preserving_rewire, er_null, weight_shuffle
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from flyopt.variants.fly_proposer import FlyProposer, FlyProposerConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_RAW = REPO_ROOT / "results" / "raw"

SUBSTRATE_KINDS = ["flywire", "er_null", "degree_preserving_null", "weight_shuffle_null"]
MAIN_SEEDS = list(range(30))
TUNING_SEEDS = [1001, 1002, 1003]
GRID_STEP_SCALE = [0.5, 1.0, 2.0]
GRID_DECODE_SCALE = [0.2, 0.5, 1.0]
GRID_READOUT_LR = [0.0]  # Faz 1/1b: frozen readout only. Faz 1c overrides this (see main()).
T_STEPS = 15
DIM = 10
N_READOUT = 50
PROPOSER_SEED = 777  # fixed: encode/decode indices + readout matrix are seed-only, never substrate-dependent (PROTOCOL.md 4.1)

# This workload is memory-bandwidth bound, not core-count bound: measured
# aggregate throughput actually DROPS going from 2 to 6 concurrent worker
# processes (isolated: 37.7s/task; 2-way parallel: 27.3s/task-equivalent;
# 6-way parallel: 29.7s/task-equivalent, worse than 2-way). More cores !=
# more throughput here — see EXPERIMENTS.md 2026-09-14 cost-overrun entry
# for the full benchmark trail (including the OOM kill that also happened
# at 12 workers). 2 workers is the measured sweet spot on this machine.
THREADS_PER_WORKER = 4

# Faz 1b (preregistration/faz_1b.md): real afferent/efferent neuron pools
# instead of "whole population" (see FlyProposer docstring). None until
# data/processed/{afferent,efferent}_indices.npy exist.
AFFERENT_POOL_PATH = DATA_PROCESSED / "afferent_indices.npy"
EFFERENT_POOL_PATH = DATA_PROCESSED / "efferent_indices.npy"


def build_weights(kind: str, seed: int, base_weights: sparse.csr_matrix) -> sparse.csr_matrix:
    if kind == "flywire":
        return base_weights
    if kind == "er_null":
        return er_null(base_weights, seed)
    if kind == "degree_preserving_null":
        return degree_preserving_rewire(base_weights, seed, n_sweeps=8)
    if kind == "weight_shuffle_null":
        return weight_shuffle(base_weights, seed)
    raise ValueError(kind)


def _load_pools(use_real_pools: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not use_real_pools:
        return None, None
    return np.load(AFFERENT_POOL_PATH), np.load(EFFERENT_POOL_PATH)


def _tune_worker(kind: str, seed: int, grid, budget: int, use_real_pools: bool = False) -> dict:
    import torch

    torch.set_num_threads(THREADS_PER_WORKER)
    base_weights = sparse.load_npz(DATA_PROCESSED / "adjacency.npz")
    weights = build_weights(kind, seed, base_weights)
    encode_pool, decode_pool = _load_pools(use_real_pools)
    out = {}
    for step_scale, decode_scale, readout_lr in grid:
        substrate = SparseRecurrentSubstrate(weights)
        config = FlyProposerConfig(
            dim=DIM, n_readout=N_READOUT, T=T_STEPS, step_scale=step_scale,
            decode_scale=decode_scale, readout_lr=readout_lr,
        )
        proposer = FlyProposer(substrate, config, seed=PROPOSER_SEED, encode_pool=encode_pool, decode_pool=decode_pool)
        best_fx = run_hillclimb(proposer, rastrigin, dim=DIM, bounds=RASTRIGIN_BOUNDS, budget=budget, seed=seed)
        out[f"{step_scale}|{decode_scale}|{readout_lr}"] = best_fx
    return {"kind": kind, "seed": seed, "results": out}


def _main_worker(
    kind: str,
    seed: int,
    step_scale: float,
    decode_scale: float,
    main_budget: int,
    sphere_budget: int,
    use_real_pools: bool = False,
    readout_lr: float = 0.0,
) -> dict:
    import torch

    torch.set_num_threads(THREADS_PER_WORKER)
    t0 = time.time()
    base_weights = sparse.load_npz(DATA_PROCESSED / "adjacency.npz")
    weights = build_weights(kind, seed, base_weights)
    null_gen_time = time.time() - t0
    encode_pool, decode_pool = _load_pools(use_real_pools)

    config = FlyProposerConfig(
        dim=DIM, n_readout=N_READOUT, T=T_STEPS, step_scale=step_scale,
        decode_scale=decode_scale, readout_lr=readout_lr,
    )

    t0 = time.time()
    substrate = SparseRecurrentSubstrate(weights)
    proposer = FlyProposer(substrate, config, seed=PROPOSER_SEED, encode_pool=encode_pool, decode_pool=decode_pool)
    rastrigin_best = run_hillclimb(
        proposer, rastrigin, dim=DIM, bounds=RASTRIGIN_BOUNDS, budget=main_budget, seed=seed
    )
    rastrigin_time = time.time() - t0

    t0 = time.time()
    substrate2 = SparseRecurrentSubstrate(weights)
    proposer2 = FlyProposer(substrate2, config, seed=PROPOSER_SEED, encode_pool=encode_pool, decode_pool=decode_pool)
    sphere_best = run_hillclimb(proposer2, sphere, dim=DIM, bounds=SPHERE_BOUNDS, budget=sphere_budget, seed=seed)
    sphere_time = time.time() - t0

    return {
        "kind": kind,
        "seed": seed,
        "step_scale": step_scale,
        "decode_scale": decode_scale,
        "readout_lr": readout_lr,
        "rastrigin_best_fx": rastrigin_best,
        "sphere_best_fx": sphere_best,
        "null_gen_time_s": null_gen_time,
        "rastrigin_time_s": rastrigin_time,
        "sphere_time_s": sphere_time,
    }


def run_faz1(
    main_budget: int = 10000,
    sphere_budget: int = 1000,
    tuning_budget: int = 300,
    seeds: list[int] = None,
    max_workers: int = 2,  # measured sweet spot: this workload is memory-bandwidth bound, not core-count bound
    log_path: Path = None,
    use_real_pools: bool = False,
    readout_lr_grid: list[float] = None,
) -> dict:
    seeds = seeds if seeds is not None else MAIN_SEEDS
    readout_lr_grid = readout_lr_grid if readout_lr_grid is not None else GRID_READOUT_LR
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    default_log_name = "faz1b_progress.jsonl" if use_real_pools else "faz1_progress.jsonl"
    log_path = log_path or (RESULTS_RAW / default_log_name)
    grid = list(itertools.product(GRID_STEP_SCALE, GRID_DECODE_SCALE, readout_lr_grid))

    def log(entry: dict):
        entry["ts"] = time.time()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    log({"event": "start", "seeds": seeds, "main_budget": main_budget, "tuning_budget": tuning_budget, "use_real_pools": use_real_pools, "readout_lr_grid": readout_lr_grid})

    # --- Phase A: equal-budget hyperparameter tuning per substrate kind ---
    best_hparams: dict[str, tuple[float, float, float]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_tune_worker, kind, seed, grid, tuning_budget, use_real_pools): (kind, seed)
            for kind in SUBSTRATE_KINDS
            for seed in TUNING_SEEDS
        }
        agg: dict[str, dict[str, list[float]]] = {
            k: {f"{s}|{d}|{r}": [] for s, d, r in grid} for k in SUBSTRATE_KINDS
        }
        for fut in as_completed(futures):
            res = fut.result()
            for key, val in res["results"].items():
                agg[res["kind"]][key].append(val)
            log({"event": "tune_seed_done", "kind": res["kind"], "seed": res["seed"]})

    for kind in SUBSTRATE_KINDS:
        means = {key: float(np.mean(vals)) for key, vals in agg[kind].items()}
        best_key = min(means, key=means.get)
        s, d, r = best_key.split("|")
        best_hparams[kind] = (float(s), float(d), float(r))
        log({"event": "tune_done", "kind": kind, "best_step_scale": float(s), "best_decode_scale": float(d), "best_readout_lr": float(r), "mean_best_fx": means[best_key]})

    # --- Phase B: main run, fixed hyperparams per substrate, all seeds ---
    raw_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for kind in SUBSTRATE_KINDS:
            step_scale, decode_scale, readout_lr = best_hparams[kind]
            for seed in seeds:
                fut = ex.submit(
                    _main_worker, kind, seed, step_scale, decode_scale, main_budget,
                    sphere_budget, use_real_pools, readout_lr,
                )
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
            "test": gate_result.test,
            "p_value": gate_result.p_value,
            "effect_size": gate_result.effect_size,
            "passes_falsification_gate": gate_pass,
            "flywire_mean": float(flywire.mean()),
            "degree_preserving_null_mean": float(degree_null.mean()),
        },
        "secondary": {
            k: {"p_value": v.p_value, "effect_size": v.effect_size, "holm_reject": holm_reject[i]}
            for i, (k, v) in enumerate(secondary.items())
        },
        "medians": {
            "flywire": float(np.median(flywire)),
            "degree_preserving_null": float(np.median(degree_null)),
            "er_null": float(np.median(er)),
            "weight_shuffle_null": float(np.median(weight_shuffle_arr)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny budgets, 2 seeds, for pipeline validation")
    parser.add_argument(
        "--faz1b", action="store_true",
        help="use real afferent/efferent neuron pools instead of whole-population sampling (preregistration/faz_1b.md)",
    )
    parser.add_argument(
        "--faz1c", action="store_true",
        help="Faz 1b pools + reward-modulated readout plasticity, tuning grid includes readout_lr (preregistration/faz_1c.md)",
    )
    args = parser.parse_args()
    use_pools = args.faz1b or args.faz1c

    if use_pools and not (AFFERENT_POOL_PATH.exists() and EFFERENT_POOL_PATH.exists()):
        raise SystemExit(
            f"--faz1b/--faz1c requires {AFFERENT_POOL_PATH} and {EFFERENT_POOL_PATH}. "
            "Build them from data/raw/Supplemental_file1_neuron_annotations.tsv first."
        )

    readout_lr_grid = [0.0, 0.05, 0.2] if args.faz1c else None
    log_name = "faz1c_progress.jsonl" if args.faz1c else ("faz1b_progress.jsonl" if args.faz1b else "faz1_progress.jsonl")
    log_path = RESULTS_RAW / log_name

    if args.smoke:
        result = run_faz1(main_budget=50, sphere_budget=20, tuning_budget=10, seeds=[0, 1], max_workers=4, use_real_pools=use_pools, readout_lr_grid=readout_lr_grid, log_path=log_path)
    else:
        result = run_faz1(main_budget=1500, sphere_budget=200, tuning_budget=100, seeds=MAIN_SEEDS, use_real_pools=use_pools, readout_lr_grid=readout_lr_grid, log_path=log_path)

    analysis = analyze(result)
    print(json.dumps(analysis, indent=2))

    phase_name = "faz1c" if args.faz1c else ("faz1b" if args.faz1b else "faz1")
    runner = ExperimentRunner(REPO_ROOT, REPO_ROOT / "results")
    record = runner.start(
        config={"phase": phase_name, "smoke": args.smoke, "best_hparams": result["best_hparams"], "main_budget": result["main_budget"]},
        seed=0,
    )
    positive = analysis["gate"]["passes_falsification_gate"]
    out_path = runner.finish(record, {"analysis": analysis, "raw_results": result["raw_results"]}, positive=positive)
    print(f"\nWrote {out_path}")
    print(f"\nGATE {'PASSED' if positive else 'FAILED'} (degree_preserving_null vs flywire, PROTOCOL.md section 2, phase={phase_name})")


if __name__ == "__main__":
    main()
