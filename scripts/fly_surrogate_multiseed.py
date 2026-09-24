"""Fly-Surrogate multi-seed pilot (2026-09-17, queued to run unattended
while the user is offline, "1 2 saatlik gorev sirala"). Real connectome
vs ER-null on a genuinely different computational role than anything
tried today: predicting the objective's VALUE at a point (regression),
not proposing a direction. Same verified-connectivity encode/decode
selection and differentiable leaky-rate substrate as the rest of
2026-09-17's pilots. Multi-seed from the start (8 seeds) so this
produces a real Wilcoxon/Cliff's-delta result, not another single-seed
"peek" -- informal, not pre-registered, but statistically honest.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse, stats

from flyopt.benchmarks import RASTRIGIN_BOUNDS
from flyopt.substrates.graph_builders import er_null
from flyopt.variants.fly_surrogate import FlySurrogate, SurrogateConfig, build_surrogate_training_set, train_surrogate
from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
DIM = 2
N_ENCODE = DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
SEEDS = list(range(8))


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_ENCODE, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=11000,
    )
    print(f"verified-connected: {len(encode_full)} encode, {len(decode_full)} decode neurons")

    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=11000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = SurrogateConfig(dim=DIM, T=8, input_scale=1.0)
    log_path = "C:/projeler/fly_op/results/fly_surrogate_multiseed.jsonl"
    results = {"real_connectome": [], "er_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_surrogate_training_set(DIM, RASTRIGIN_BOUNDS, n_problems=15, n_points=20, seed=20000 + seed)
            X_test, Y_test = build_surrogate_training_set(DIM, RASTRIGIN_BOUNDS, n_problems=8, n_points=20, seed=30000 + seed)

            sub_null = er_null(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                brain = FlySurrogate(sub, encode_idx, decode_idx, cfg, seed=seed)
                train_surrogate(brain, X, Y, epochs=EPOCHS, lr=LR)

                import torch
                import torch.nn.functional as F
                y_mean, y_std = float(Y.mean()), float(Y.std() + 1e-6)
                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test))
                    y_test_norm = torch.as_tensor((Y_test - y_mean) / y_std)
                    test_mse = float(F.mse_loss(pred, y_test_norm).item())

                results[name].append(test_mse)
                rec = {"seed": seed, "substrate": name, "test_mse": test_mse}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] test_mse={test_mse:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["er_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    gt = np.sum(real[:, None] < null[None, :])  # lower mse = better, so "wins" = real < null
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print("\n=== SONUC (Fly-Surrogate, n=8 tohum) ===")
    print(f"real_connectome test_mse: {real.tolist()}")
    print(f"er_null         test_mse: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f} (pozitif = gercek daha iyi/dusuk mse)")
    print(f"gate (p<0.05 and |delta|>0.33): {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_surrogate_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "real_connectome": real.tolist(), "er_null": null.tolist(),
            "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta,
        }, f, indent=2)
    print("wrote results/fly_surrogate_summary.json")


if __name__ == "__main__":
    main()
