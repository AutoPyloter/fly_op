"""First-ever test of training real connectome synapses (PROTOCOL.md 2.5
Fly-RL, never run before -- see EXPERIMENTS.md 2026-09-17, "biz
connectome'un kendi agirligini egittigimiz bir calisma oldu mu?" /
"kucuk olcekte dene, birkac yuz sinapsla basla"). Informal, not
pre-registered.

A FIXED, already-calibrated readout (fly_proposer_scene.calibrate_readout,
single landscape) is the baseline. Then N_SYNAPSES=300 real FlyWire
synapse weights are perturbed by a (1+1) hill-climb (search_loop.py's own
algorithm, applied to the connectome's own parameters this time) to
minimize the SAME readout's prediction error on a held-out multi-problem
batch. Train/test split: the hill-climb only ever sees `train_batch`;
`test_batch` (disjoint seed) is scored once at the end for an honest
generalization check, not just curve-fitting the search batch.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from flyopt.benchmarks import RASTRIGIN_BOUNDS, rastrigin
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from flyopt.variants.fly_proposer_scene import FlyProposerScene, SceneProposerConfig, calibrate_readout
from flyopt.variants.synapse_plasticity import build_eval_batch, evaluate, hillclimb_synapses, perturbed_weights

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DIM = 2
N_SYNAPSES = 300
BUDGET = 150
SIGMA = 0.3


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

    train_batch = build_eval_batch(DIM, RASTRIGIN_BOUNDS, n_problems=4, n_starts=4, seed=5000)
    test_batch = build_eval_batch(DIM, RASTRIGIN_BOUNDS, n_problems=6, n_starts=6, seed=6000)  # disjoint seed, held out

    baseline_train_mse = evaluate(base_weights, template, train_batch)
    baseline_test_mse = evaluate(base_weights, template, test_batch)
    print(f"baseline (real connectome, unperturbed): train_mse={baseline_train_mse:.4f}  test_mse={baseline_test_mse:.4f}")

    theta, synapse_idx, curve = hillclimb_synapses(
        base_weights, template, n_synapses=N_SYNAPSES, budget=BUDGET, sigma=SIGMA, batch=train_batch, seed=42,
    )

    trained_weights = perturbed_weights(base_weights.tocoo(), synapse_idx, theta)
    trained_test_mse = evaluate(trained_weights, template, test_batch)
    print(f"\n=== SONUC ===")
    print(f"baseline test_mse (gercek connectome, {N_SYNAPSES} sinaps degismedi): {baseline_test_mse:.4f}")
    print(f"trained  test_mse ({N_SYNAPSES} sinaps {BUDGET} iterasyon egitildi):    {trained_test_mse:.4f}")
    improvement = (baseline_test_mse - trained_test_mse) / baseline_test_mse * 100
    print(f"iyilesme: {improvement:+.1f}%  (pozitif = egitim yardimci oldu, negatif = zarar verdi/asiri uyum)")

    import json
    with open("C:/projeler/fly_op/results/fly_synapse_plasticity.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_synapses": N_SYNAPSES, "budget": BUDGET, "sigma": SIGMA,
            "baseline_train_mse": baseline_train_mse, "baseline_test_mse": baseline_test_mse,
            "trained_test_mse": trained_test_mse, "improvement_pct": improvement,
            "train_curve": curve,
        }, f, indent=2)
    print("wrote results/fly_synapse_plasticity.json")


if __name__ == "__main__":
    main()
