"""n=8 pilot (fly_mackey_glass_prediction_multiseed.py) crossed the gate
(Cliff's delta=-0.719, Mann-Whitney p=0.0148) -- again in the reversed
direction (real connectome WORSE, higher test MSE, than the
degree-preserving null). Extends to n=30 with 22 new independently-
generated Mackey-Glass series (seeds 8-29), combined with 0-7.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.rate_brain import (
    RateBrain,
    RateBrainConfig,
    build_subgraph_bfs,
    select_connected_encode_decode,
)
from fly_mackey_glass_prediction_multiseed import (
    DATA_PROCESSED, N_ENCODE, N_READOUT, SUBGRAPH_SIZE, WINDOW, EPOCHS, LR, SERIES_LEN,
    generate_mackey_glass, windows_from_series, forward_sequence, train_sequence, DEVICE,
)

SEEDS = list(range(8, 30))


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_ENCODE, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = RateBrainConfig(dim=1, n_readout=len(decode_idx), T=WINDOW, decode_scale=0.5, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_mackey_glass_prediction_seeds8to29.jsonl"
    results = {"real": [], "null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            series = generate_mackey_glass(SERIES_LEN, seed=30000 + seed)
            split = int(len(series) * 0.7)
            X_all, Y_all = windows_from_series(series)
            n_train = split - WINDOW
            X_train, Y_train = X_all[:n_train], Y_all[:n_train]
            X_test, Y_test = X_all[n_train:], Y_all[n_train:]
            X_test_t = torch.as_tensor(X_test, device=DEVICE)
            Y_test_t = torch.as_tensor(Y_test, device=DEVICE)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            row = {"seed": seed}
            for name, sub in [("real", sub_real), ("null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train_sequence(brain, X_train, Y_train, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    pred = forward_sequence(brain, X_test_t)
                    test_mse = float(((pred - Y_test_t) ** 2).mean().item())
                results[name].append(test_mse)
                row[f"{name}_test_mse"] = test_mse
                del brain
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed={seed}: {row}")

    orig = [json.loads(l) for l in open("C:/projeler/fly_op/results/fly_mackey_glass_prediction_multiseed.jsonl", encoding="utf-8")]
    real = np.array([r["real_test_mse"] for r in orig] + results["real"])
    null = np.array([r["null_test_mse"] for r in orig] + results["null"])
    assert len(real) == 30 and len(null) == 30

    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33

    print("\n=== SONUC (Mackey-Glass, n=30) ===")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Mann-Whitney p={mw_p:.6f}  Wilcoxon p={w_p:.6f}  Cliff's delta={cliffs_delta:.3f}  gate={'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_mackey_glass_prediction_official30_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate)}, f, indent=2)
    print("wrote results/fly_mackey_glass_prediction_official30_summary.json")


if __name__ == "__main__":
    main()
