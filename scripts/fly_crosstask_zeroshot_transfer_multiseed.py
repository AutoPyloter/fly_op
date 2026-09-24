"""Cross-task zero-shot transfer test (2026-09-23/24, overnight autonomous
battery). Trains real vs null RateBrain on slope-stability ONLY (identical
recipe to the official n=30 result), then evaluates the SAME trained
weights -- no further training, no re-selection of encode/decode neurons --
directly on the beam-design task. Tests whether the "single-shot direction
finding" skill transfers zero-shot across physical task types on the same
substrate, or is narrowly overfit to the specific task it was trained on.

Dimensionality note: slope-stability is DIM=3 (encode=6 rays, readout=3),
beam-design is DIM=2 (b, h). To reuse the identical trained network
(same encode/decode indices, same readout_W shape) with NO architecture
change, beam points are embedded in the same 3D space with a dummy third
coordinate that beam_cost ignores (so its ray/teacher-target component is
always exactly 0 by construction, not a hack that leaks information) --
the zero-shot metric compares only the first 2 predicted dimensions
(b, h) against the true beam-design teacher target.

n=8 seeds (each = one slope-trained network pair, evaluated zero-shot on
a fresh beam eval set). Same null-comparison protocol as everywhere else.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM as SLOPE_DIM, factor_of_safety, random_geometry as slope_random_geometry
from flyopt.benchmarks_structural import beam_cost, random_geometry as beam_random_geometry, sample_valid_point as beam_sample_valid_point
from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
    train,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * SLOPE_DIM  # 6, dictated by the slope-trained network's fixed architecture
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
SLOPE_RAY_RADIUS = 1.0
BEAM_RAY_RADIUS = 0.02
SEEDS = list(range(8))


def build_slope_training_set(n_problems: int, n_starts: int, seed: int):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = slope_random_geometry(rng)
        lo, hi = geo.param_bounds()
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi)
            fx = f(x)
            rays = _rays(x, f, fx, SLOPE_RAY_RADIUS)
            target = _teacher_delta(x, rays, SLOPE_RAY_RADIUS)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(SLOPE_DIM)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def build_beam_padded_eval_set(n_problems: int, n_starts: int, seed: int):
    """Beam-design (b, h) embedded in 3D with a dummy 3rd coordinate that
    beam_cost ignores entirely -- keeps the encode/decode dimensionality
    identical to the slope-trained network with no leakage."""
    rng = np.random.default_rng(seed)

    def f3(x3, geo):
        return beam_cost(x3[:2], geo)

    X, Y = [], []
    for _ in range(n_problems):
        geo = beam_random_geometry(rng)
        for _ in range(n_starts):
            x2 = beam_sample_valid_point(geo, rng)
            x3 = np.concatenate([x2, [0.0]])
            fx = f3(x3, geo)
            rays = _rays(x3, lambda p: f3(p, geo), fx, BEAM_RAY_RADIUS)
            target = _teacher_delta(x3, rays, BEAM_RAY_RADIUS)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(SLOPE_DIM)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = RateBrainConfig(dim=SLOPE_DIM, n_readout=len(decode_idx), T=8, ray_radius=SLOPE_RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_crosstask_zeroshot_transfer_multiseed.jsonl"
    results = {"real": [], "null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_slope_training_set(n_problems=20, n_starts=10, seed=9500 + seed)
            X_beam, Y_beam = build_beam_padded_eval_set(n_problems=10, n_starts=10, seed=29500 + seed)
            X_beam_t = torch.as_tensor(X_beam, device=DEVICE)
            Y_beam_t = torch.as_tensor(Y_beam, device=DEVICE)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed}
            for name, sub in [("real", sub_real), ("null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)  # trained on SLOPE ONLY
                with torch.no_grad():
                    pred = brain(X_beam_t)  # zero-shot, no beam training
                    zeroshot_mse = float(((pred[:, :2] - Y_beam_t[:, :2]) ** 2).mean().item())
                results[name].append(zeroshot_mse)
                row[f"{name}_zeroshot_beam_mse"] = zeroshot_mse
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed}: {row}")

    real = np.array(results["real"])
    null = np.array(results["null"])
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])  # lower zero-shot mse = better transfer
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33

    print("\n=== SONUC (capraz-gorev sifir-atis transfer: slope-egitimli -> beam-degerlendirme, n=8) ===")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_crosstask_zeroshot_transfer_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_crosstask_zeroshot_transfer_summary.json")


if __name__ == "__main__":
    main()
