"""Does the fly connectome's advantage require ITS OWN hub structure, or
does ANY heavy-tailed/hub-dominated graph do just as well? (2026-09-20,
user: "sentetik hub-baskin grafla da test et", following the observation
that `degree_preserving_rewire` -- which reuses the fly's EXACT degree
sequence but scrambles who-connects-to-whom -- still wins strongly
(delta=0.884, n=30), while `weight_shuffle` -- which keeps the fly's
EXACT real topology but scrambles weight values -- barely shows an
effect (delta=0.258, n=30, FAIL). That pattern already suggested the
effect is about degree heterogeneity, not fly-specific wiring identity.
This is the direct test: `scale_free_correlated_null` (graph_builders.py) samples a
FRESH power-law degree sequence, unrelated to the fly's own, matched
only on node count and total edge count -- if this null also loses to
the real connectome, the advantage is generic to the "heavy-tailed
degree" topology CLASS, not the fly's connectome specifically.

Same architecture/training/eval recipe as
fly_slope_stability_degreenull_official30.py, same subgraph
(encode-seed=9000). Since the synthetic graph's topology is unrelated to
the real one, encode/decode nodes are re-selected (verified
forward-connected) SEPARATELY on the synthetic graph each seed, rather
than reusing the real graph's encode/decode indices -- reusing them
would unfairly handicap the synthetic condition if a real high-out-degree
encode neuron happens to land on a low-degree node in the fresh
Pareto-sampled assignment.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry
from flyopt.substrates.graph_builders import scale_free_correlated_null
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
    train,
)

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(8))


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


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    all_nodes = np.arange(SUBGRAPH_SIZE)

    log_path = "C:/projeler/fly_op/results/fly_slope_stability_scalefreecorr_multiseed.jsonl"
    results = {"real_connectome": [], "scale_free_correlated_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_geo(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_geo(n_problems=10, n_starts=10, seed=19500 + seed)
            sub_synth = scale_free_correlated_null(sub_real, seed=seed)

            synth_encode_full, synth_decode_full = select_connected_encode_decode(
                sub_synth, all_nodes, all_nodes, n_encode=N_RAYS, n_decode=N_READOUT,
                max_hops=6, n_encode_candidates=200, seed=20000 + seed,
            )
            coo = sub_synth.tocoo()
            print(f"seed={seed} scale_free_correlated_null: {sub_synth.shape[0]} nodes, {coo.nnz} edges, "
                  f"{len(synth_encode_full)} verified encode / {len(synth_decode_full)} verified decode")

            for name, sub, e_idx, d_idx in [
                ("real_connectome", sub_real, encode_idx, decode_idx),
                ("scale_free_correlated_null", sub_synth, synth_encode_full, synth_decode_full[:N_READOUT]),
            ]:
                if len(d_idx) < 5:
                    raise RuntimeError(f"seed={seed} {name}: only {len(d_idx)} decode neurons reachable -- widen max_hops/candidates")
                cfg = RateBrainConfig(dim=DIM, n_readout=len(d_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=0.5, train_gain=True)
                brain = RateBrain(sub, e_idx, d_idx, cfg, seed=seed)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)

                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test))
                    test_mse = float(((pred - torch.as_tensor(Y_test)) ** 2).mean().item())

                results[name].append(test_mse)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": test_mse}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] test_mse={test_mse:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["scale_free_correlated_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    mw_stat, mw_pvalue = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (sev stabilitesi, sentetik KORELE hub-baskin (scale-free) null, n=8 tohum) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Mann-Whitney p={mw_pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_slope_stability_scalefreecorr_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(pvalue),
                   "mannwhitney_p": float(mw_pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_slope_stability_scalefreecorr_summary.json")


if __name__ == "__main__":
    main()
