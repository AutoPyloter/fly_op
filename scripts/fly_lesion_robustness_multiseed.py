"""Structural-damage tolerance test (2026-09-23/24, overnight autonomous
battery -- user: "hepsini dene olc bic degerlendir raporla ... sabaha kadar
calis"). Tests a genuinely different axis than every prior test: not "can
the trained network solve the task better", but "once trained, how
gracefully does it degrade when part of its wiring is destroyed at
inference time (no retraining)". A structural-robustness advantage would
NOT be visible in any of the task-performance gates already run.

Same slope-stability single-shot task/training recipe as the official
result (fly_slope_stability_degreenull_official30.py: T=8, 300 epochs,
lr=3e-3, decode_scale=0.5, 3000-node BFS subgraph, degree_preserving_rewire
null). After training, zero out a random X% of the RECURRENT edges
in-place (5 independent ablation draws averaged per network, to reduce
per-draw noise) at X in {10, 20, 30}, re-evaluate on the same held-out
test set with NO further training, and compare the degradation ratio
(ablated_mse / baseline_mse -- lower is more robust) between real and
degree_preserving_rewire null. Official gate test is at X=20%, n=8 seeds.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
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
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(8))
ABLATION_FRACS = [0.10, 0.20, 0.30]
GATE_FRAC = 0.20
N_ABLATION_DRAWS = 5


def build_training_set_geo(n_problems: int, n_starts: int, seed: int):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_problems):
        geo = random_geometry(rng)
        lo, hi = geo.param_bounds()
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi)
            fx = f(x)
            rays = _rays(x, f, fx, RAY_RADIUS)
            target = _teacher_delta(x, rays, RAY_RADIUS)
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(DIM)
            X.append(rays)
            Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def eval_mse(brain, X_t, Y_t) -> float:
    with torch.no_grad():
        pred = brain(X_t)
        return float(((pred - Y_t) ** 2).mean().item())


def ablated_mse(brain, X_t, Y_t, frac: float, rng: np.random.Generator, n_draws: int) -> float:
    """Zero out `frac` of the recurrent edges (in-place on the `sign`
    buffer, restored after each draw), average MSE over n_draws
    independent random masks. Inference only -- no gradient, no retraining."""
    nnz = brain.sign.shape[0]
    n_zero = int(round(frac * nnz))
    original = brain.sign.clone()
    mses = []
    for _ in range(n_draws):
        idx = rng.choice(nnz, size=n_zero, replace=False)
        brain.sign[idx] = 0.0
        mses.append(eval_mse(brain, X_t, Y_t))
        brain.sign.copy_(original)
    return float(np.mean(mses))


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

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_lesion_robustness_multiseed.jsonl"
    per_frac_real = {f: [] for f in ABLATION_FRACS}
    per_frac_null = {f: [] for f in ABLATION_FRACS}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            X_test_t = torch.as_tensor(X_test, device=DEVICE)
            Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed}
            for name, sub, store in [("real", sub_real, per_frac_real), ("null", sub_null, per_frac_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)
                baseline = eval_mse(brain, X_test_t, Y_test_t)
                ablate_rng = np.random.default_rng(70000 + seed)
                row[f"{name}_baseline"] = baseline
                for frac in ABLATION_FRACS:
                    abl_mse = ablated_mse(brain, X_test_t, Y_test_t, frac, ablate_rng, N_ABLATION_DRAWS)
                    ratio = abl_mse / baseline if baseline > 1e-9 else float("nan")
                    store[frac].append(ratio)
                    row[f"{name}_ablmse_{int(frac*100)}"] = abl_mse
                    row[f"{name}_ratio_{int(frac*100)}"] = ratio
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "seed"))

    summary = {}
    for frac in ABLATION_FRACS:
        real = np.array(per_frac_real[frac])
        null = np.array(per_frac_null[frac])
        w_stat, w_p = stats.wilcoxon(real, null)
        mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
        gt = np.sum(real[:, None] < null[None, :])  # lower degradation ratio = more robust
        lt = np.sum(real[:, None] > null[None, :])
        cliffs_delta = float((gt - lt) / (len(real) * len(null)))
        gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33
        summary[str(int(frac * 100))] = {
            "real": real.tolist(), "null": null.tolist(),
            "real_median": float(np.median(real)), "null_median": float(np.median(null)),
            "wilcoxon_p": float(w_p), "mannwhitney_p": float(mw_p),
            "cliffs_delta": cliffs_delta, "gate_pass": bool(gate),
        }
        star = " <== RESMI KAPI TESTI" if frac == GATE_FRAC else ""
        print(f"\n=== ablation={int(frac*100)}% (degradation ratio, dusuk=daha saglam) ==={star}")
        print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
        print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}  gate={'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_lesion_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote results/fly_lesion_robustness_summary.json")
    print(f"RESMI KAPI (X=20%): {'PASS' if summary['20']['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
