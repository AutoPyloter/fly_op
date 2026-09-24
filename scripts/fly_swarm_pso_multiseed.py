"""Multi-seed robustness check for fly_swarm_pso_demo.py's single-run result
(scene_anneal found the true optimum, others got stuck at a local one).
Still not a pre-registered phase -- informal robustness check requested
2026-09-17 ("cok tohum yap bakalim") before deciding whether this is worth
turning into a real Faz.

Flies are built (and calibrated, for scene/scene_anneal) ONCE per variant,
then reused across N_SEEDS swarm trials with persistent state cleared
between trials (fly_swarm_pso_demo.reset_flies) -- calibration doesn't
depend on the swarm's own seed, only on each particle's own construction
seed, so recalibrating per trial would be wasted compute.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from fly_swarm_pso_demo import DATA_PROCESSED, make_flies, run_variant

N_SEEDS = 8
SEEDS = list(range(N_SEEDS))
VARIANTS = ["classic", "raw", "scene", "scene_anneal"]


def main():
    weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    substrate = SparseRecurrentSubstrate(weights)

    # scene and scene_anneal share the same calibrated flies -- fly_weight
    # scheduling lives in run_variant's loop, not in the proposer itself.
    flies_by_variant = {
        "classic": None,
        "raw": make_flies("raw", substrate),
        "scene": make_flies("scene", substrate),
    }
    flies_by_variant["scene_anneal"] = flies_by_variant["scene"]

    results: dict[str, list[float]] = {v: [] for v in VARIANTS}
    log_path = "C:/projeler/fly_op/results/fly_swarm_pso_multiseed.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            for variant in VARIANTS:
                _, _, gbest_x = run_variant(variant, substrate, seed=seed, flies=flies_by_variant[variant])
                fx = float(np.sum(np.asarray(gbest_x) ** 2 - 10 * np.cos(2 * np.pi * np.asarray(gbest_x))) + 20)
                results[variant].append(fx)
                rec = {"seed": seed, "variant": variant, "gbest_fx": fx, "gbest_x": list(map(float, gbest_x))}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"seed={seed} [{variant:14s}] gbest_fx={fx:8.4f}")

    print("\n=== ozet (n=%d tohum) ===" % N_SEEDS)
    for variant in VARIANTS:
        vals = np.array(results[variant])
        n_solved = int(np.sum(vals < 0.1))  # arbitrary "found the real optimum" threshold
        print(f"{variant:14s} mean={vals.mean():8.4f} median={np.median(vals):8.4f} "
              f"min={vals.min():8.4f} max={vals.max():8.4f}  gercek_minimumu_buldu={n_solved}/{N_SEEDS}")


if __name__ == "__main__":
    main()
