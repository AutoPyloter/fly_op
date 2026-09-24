"""Synapse plasticity v2 (2026-09-17, "biraz daha detayli ara"): fixes the
two weaknesses flagged after v1's null result (EXPERIMENTS.md 2026-09-17,
fly_synapse_plasticity_demo.py) -- targeted synapse selection instead of
uniformly random, and an adaptive-sigma block-coordinate search instead of
a fixed-sigma full-dimensional (1+1). Same readout-frozen isolation logic,
same train/test split discipline. Informal, not pre-registered.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from flyopt.benchmarks import RASTRIGIN_BOUNDS, rastrigin
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from flyopt.variants.fly_proposer_scene import FlyProposerScene, SceneProposerConfig, calibrate_readout
from flyopt.variants.synapse_plasticity import (
    build_eval_batch,
    evaluate,
    hillclimb_synapses,
    perturbed_weights,
    select_targeted_synapses,
)

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DIM = 2
N_SYNAPSES = 300
BUDGET = 400
SIGMA = 0.3
BLOCK_FRAC = 0.15
ADAPT_WINDOW = 15


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    substrate = SparseRecurrentSubstrate(base_weights)
    encode_pool = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    decode_pool = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    cfg = SceneProposerConfig(dim=DIM, n_readout=50, T=15, ray_radius=0.3, ray_scale=1.0, decode_scale=0.5)
    template = FlyProposerScene(substrate, cfg, seed=4000, encode_pool=encode_pool, decode_pool=decode_pool, reset_seed=0)
    template.bind_objective(rastrigin)
    calibrate_readout(template, rastrigin, dim=DIM, bounds=RASTRIGIN_BOUNDS, n_steps=100, seed=4000)
    print("readout calibrated (baseline, frozen for the rest of this experiment)")

    synapse_idx = select_targeted_synapses(
        base_weights, template.encode_indices, template.decode_indices, n_synapses=N_SYNAPSES, seed=42,
    )
    print(f"selected {len(synapse_idx)} synapses directly touching the encode/decode neurons "
          f"(out of {base_weights.nnz:,} total)")

    train_batch = build_eval_batch(DIM, RASTRIGIN_BOUNDS, n_problems=4, n_starts=4, seed=5000)
    test_batch = build_eval_batch(DIM, RASTRIGIN_BOUNDS, n_problems=6, n_starts=6, seed=6000)

    baseline_train_mse = evaluate(base_weights, template, train_batch)
    baseline_test_mse = evaluate(base_weights, template, test_batch)
    print(f"baseline (real connectome, unperturbed): train_mse={baseline_train_mse:.4f}  test_mse={baseline_test_mse:.4f}")

    theta, synapse_idx, curve = hillclimb_synapses(
        base_weights, template, synapse_idx, budget=BUDGET, sigma=SIGMA, batch=train_batch, seed=42,
        block_frac=BLOCK_FRAC, adapt_window=ADAPT_WINDOW,
    )

    trained_weights = perturbed_weights(base_weights.tocoo(), synapse_idx, theta)
    trained_test_mse = evaluate(trained_weights, template, test_batch)
    print(f"\n=== SONUC (v2) ===")
    print(f"baseline test_mse (gercek connectome, {N_SYNAPSES} sinaps degismedi): {baseline_test_mse:.4f}")
    print(f"trained  test_mse ({N_SYNAPSES} hedefli sinaps, {BUDGET} iterasyon, adaptif sigma): {trained_test_mse:.4f}")
    improvement = (baseline_test_mse - trained_test_mse) / baseline_test_mse * 100
    n_nonzero_theta = int(np.sum(np.abs(theta) > 1e-9))
    print(f"iyilesme: {improvement:+.1f}%  (pozitif = egitim yardimci oldu)")
    print(f"theta'da sifirdan farkli sinaps sayisi: {n_nonzero_theta}/{N_SYNAPSES}")

    with open("C:/projeler/fly_op/results/fly_synapse_plasticity_v2.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_synapses": N_SYNAPSES, "budget": BUDGET, "sigma": SIGMA, "block_frac": BLOCK_FRAC,
            "baseline_train_mse": baseline_train_mse, "baseline_test_mse": baseline_test_mse,
            "trained_test_mse": trained_test_mse, "improvement_pct": improvement,
            "n_nonzero_theta": n_nonzero_theta, "train_curve": curve,
        }, f, indent=2)
    print("wrote results/fly_synapse_plasticity_v2.json")


if __name__ == "__main__":
    main()
