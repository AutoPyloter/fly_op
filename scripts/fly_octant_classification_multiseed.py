"""Single-shot CLASSIFICATION benchmark (2026-09-22, collaborative
follow-up after the saccadic-navigation negative result: redirect effort
toward the task family that actually works -- single-shot, one decisive
call -- but test a genuinely different COMPUTATION TYPE than the
continuous-direction regression tasks tried so far).

Reuses the exact same validated machinery as the official slope-stability
finding (delta=0.884, n=30): the same ray/finite-difference sensory
encoding (`_rays`), the same slope-stability landscape
(`benchmarks_geo.py`), the same RateBrain architecture and subgraph
construction. The only change is the OUTPUT: instead of predicting a
continuous escape direction, the network must classify which of the
2**DIM "octants" (sign pattern of the true negative-gradient direction,
e.g. for DIM=3: +++, ++-, +-+, ..., 8 classes) the correct move belongs
to. Chance level = 1/8 = 12.5%. This requires real computation from the
ray inputs (not a trivial copy), is genuinely single-shot (no
trajectory/memory), and tests a qualitatively different task type
(discrete classification, cross-entropy) than every prior single-shot
result (continuous regression, MSE).
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse, stats

from flyopt.benchmarks_geo import DIM, factor_of_safety, random_geometry, sample_valid_point
from flyopt.substrates.graph_builders import degree_preserving_rewire
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta
from flyopt.variants.rate_brain import RateBrain, RateBrainConfig, build_subgraph_bfs, select_connected_encode_decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
N_RAYS = 2 * DIM
N_READOUT = 30
SUBGRAPH_SIZE = 3000
N_CLASSES = 2 ** DIM  # octant of the true negative-gradient direction
EPOCHS = 300
LR = 3e-3
RAY_RADIUS = 1.0
SEEDS = list(range(8))


def octant_label(target: np.ndarray) -> int:
    """Sign pattern of the (non-degenerate) teacher direction -> integer class in [0, 2**DIM)."""
    bits = (target > 0).astype(np.int64)
    return int(sum(b << i for i, b in enumerate(bits)))


def build_training_set_octant(n_problems: int, n_starts: int, seed: int):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    while len(X) < n_problems * n_starts:
        geo = random_geometry(rng)
        lo, hi = geo.param_bounds()
        f = lambda p, geo=geo: factor_of_safety(p, geo)
        for _ in range(n_starts):
            x = sample_valid_point(geo, rng)
            fx = f(x)
            rays = _rays(x, f, fx, RAY_RADIUS)
            target = _teacher_delta(x, rays, RAY_RADIUS)
            if np.any(target == 0):  # degenerate sign, skip (ambiguous octant)
                continue
            X.append(rays)
            Y.append(octant_label(target))
    return np.asarray(X[:n_problems * n_starts], dtype=np.float32), np.asarray(Y[:n_problems * n_starts], dtype=np.int64)


def train_classifier(brain, X, Y, epochs, lr):
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    device = brain.sign.device
    X_t = torch.as_tensor(X, device=device)
    Y_t = torch.as_tensor(Y, device=device)
    for epoch in range(epochs):
        opt.zero_grad()
        logits = brain(X_t)
        loss = F.cross_entropy(logits, Y_t)
        loss.backward()
        opt.step()
        if epoch % 60 == 0 or epoch == epochs - 1:
            acc = (logits.argmax(dim=1) == Y_t).float().mean().item()
            print(f"    epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}  train_acc={acc:.3f}")


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=N_RAYS, n_decode=N_READOUT,
        max_hops=6, n_encode_candidates=200, seed=9000,
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=9000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges, {len(encode_idx)} encode / {len(decode_idx)} decode, N_CLASSES={N_CLASSES}")

    cfg = RateBrainConfig(dim=N_CLASSES, n_readout=len(decode_idx), T=8, ray_radius=RAY_RADIUS, decode_scale=1.0, train_gain=True)

    log_path = "C:/projeler/fly_op/results/fly_octant_classification_multiseed.jsonl"
    results = {"real_connectome": [], "degree_preserving_null": []}
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set_octant(n_problems=20, n_starts=10, seed=9500 + seed)
            X_test, Y_test = build_training_set_octant(n_problems=10, n_starts=10, seed=19500 + seed)
            sub_null = degree_preserving_rewire(sub_real, seed=seed)

            for name, sub in [("real_connectome", sub_real), ("degree_preserving_null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed).to(DEVICE)
                train_classifier(brain, X, Y, epochs=EPOCHS, lr=LR)
                with torch.no_grad():
                    logits = brain(torch.as_tensor(X_test, device=DEVICE))
                    test_acc = float((logits.argmax(dim=1) == torch.as_tensor(Y_test, device=DEVICE)).float().mean().item())
                results[name].append(test_acc)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_acc": test_acc}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:24s}] test_acc={test_acc:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["degree_preserving_null"])
    w_stat, w_p = stats.wilcoxon(real, null)
    mw_stat, mw_p = stats.mannwhitneyu(real, null, alternative="two-sided")
    # HIGHER accuracy = better, opposite convention from the MSE tasks
    gt = np.sum(real[:, None] > null[None, :])
    lt = np.sum(real[:, None] < null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))
    gate = mw_p < 0.05 and abs(cliffs_delta) > 0.33

    print(f"\n=== SONUC (oktant siniflandirma, {N_CLASSES} sinif, sans seviyesi={1/N_CLASSES:.3f}, degree_preserving null, n=8) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={w_p:.5f}  Mann-Whitney p={mw_p:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if gate else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_octant_classification_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(w_p),
                   "mannwhitney_p": float(mw_p), "cliffs_delta": cliffs_delta, "gate_pass": bool(gate),
                   "n_classes": N_CLASSES, "chance_level": 1/N_CLASSES}, f, indent=2)
    print("wrote results/fly_octant_classification_summary.json")


if __name__ == "__main__":
    main()
