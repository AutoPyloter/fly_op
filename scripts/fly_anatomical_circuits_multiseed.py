"""Real anatomical circuits instead of generic afferent/efferent pools
(2026-09-18, "hepsini dene" -- user's brainstorm follow-up). Uses actual
FlyWire cell-type annotations (Supplemental_file1_neuron_annotations.tsv,
`cell_class` column) instead of arbitrary large sensory/motor pools:

  mushroom_body : encode=Kenyon_Cell (5,177), decode=MBON (96)
                  -- the fly's own associative-memory/learning circuit
  antennal_lobe : encode=olfactory (2,281), decode=ALPN (685)
                  -- the real chemotaxis/gradient-following circuit
  visual_hierarchy : encode=lamina (313, first visual relay),
                  decode=lobula_plate (711, motion-integrated output)
                  -- lets the REAL multi-stage visual pathway
                  (LA -> ME -> LO -> LOP) do coarse-to-fine processing,
                  instead of us hand-designing multi-scale externally

Same task (multi-problem Rastrigin direction regression), same
differentiable rate-brain substrate, same connectivity verification, same
real-vs-er_null 8-seed comparison as every other 2026-09-17/18 pilot --
only the encode/decode POOLS change, to test whether anatomically
appropriate circuits do better than the generic pools used all night.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse, stats

from flyopt.benchmarks import RASTRIGIN_BOUNDS
from flyopt.substrates.graph_builders import er_null
from flyopt.variants.rate_brain import RateBrain, RateBrainConfig, build_subgraph_bfs, build_training_set, select_connected_encode_decode, train

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
CELLTYPES = f"{DATA_PROCESSED}/celltypes"
DIM = 2
N_RAYS = 2 * DIM
N_READOUT_CAP = 30  # cap decode pool size for cost parity with other pilots
SUBGRAPH_SIZE = 3000
EPOCHS = 300
LR = 3e-3
SEEDS = list(range(8))

CIRCUITS = {
    "mushroom_body": ("kenyon_cell", "mbon"),
    "antennal_lobe": ("olfactory", "alpn"),
    "visual_hierarchy": ("lamina", "lobula_plate"),
}


def run_circuit(circuit_name: str, encode_pool_name: str, decode_pool_name: str):
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    encode_pool = np.load(f"{CELLTYPES}/{encode_pool_name}_indices.npy")
    decode_pool = np.load(f"{CELLTYPES}/{decode_pool_name}_indices.npy")
    print(f"\n=== {circuit_name}: encode={encode_pool_name} (n={len(encode_pool)}), "
          f"decode={decode_pool_name} (n={len(decode_pool)}) ===")

    n_decode = min(N_READOUT_CAP, len(decode_pool))
    encode_full, decode_full = select_connected_encode_decode(
        base_weights, encode_pool, decode_pool, n_encode=N_RAYS, n_decode=n_decode,
        max_hops=6, n_encode_candidates=min(200, len(encode_pool)), seed=14000,
    )
    print(f"verified-connected: {len(encode_full)} encode, {len(decode_full)} decode")

    sub_real, encode_idx, decode_idx, _nodes = build_subgraph_bfs(base_weights, encode_full, decode_full, SUBGRAPH_SIZE, seed=14000)
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = RateBrainConfig(dim=DIM, n_readout=len(decode_idx), T=8, ray_radius=0.3, decode_scale=0.5, train_gain=True)
    log_path = f"C:/projeler/fly_op/results/fly_circuit_{circuit_name}_multiseed.jsonl"
    results = {"real_connectome": [], "er_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            X, Y = build_training_set(DIM, RASTRIGIN_BOUNDS, n_problems=25, n_starts=12, ray_radius=0.3, seed=15000 + seed)
            X_test, Y_test = build_training_set(DIM, RASTRIGIN_BOUNDS, n_problems=10, n_starts=10, ray_radius=0.3, seed=25000 + seed)
            sub_null = er_null(sub_real, seed=seed)

            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                brain = RateBrain(sub, encode_idx, decode_idx, cfg, seed=seed)
                train(brain, X, Y, epochs=EPOCHS, lr=LR)

                import torch
                with torch.no_grad():
                    pred = brain(torch.as_tensor(X_test))
                    test_mse = float(((pred - torch.as_tensor(Y_test)) ** 2).mean().item())

                results[name].append(test_mse)
                f.write(json.dumps({"seed": seed, "substrate": name, "test_mse": test_mse}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] test_mse={test_mse:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["er_null"])
    wstat, pvalue = stats.wilcoxon(real, null)
    gt = np.sum(real[:, None] < null[None, :])
    lt = np.sum(real[:, None] > null[None, :])
    cliffs_delta = float((gt - lt) / (len(real) * len(null)))

    print(f"\n=== SONUC ({circuit_name}) ===")
    print(f"real median={np.median(real):.4f}  null median={np.median(null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open(f"C:/projeler/fly_op/results/fly_circuit_{circuit_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist(), "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    return pvalue, cliffs_delta


def main():
    summary = {}
    for circuit_name, (enc, dec) in CIRCUITS.items():
        p, delta = run_circuit(circuit_name, enc, dec)
        summary[circuit_name] = {"p": p, "delta": delta}

    print("\n\n=== TUM DEVRELER OZET ===")
    for name, r in summary.items():
        print(f"{name:20s} p={r['p']:.4f}  delta={r['delta']:+.3f}  "
              f"gate={'PASS' if (r['p'] < 0.05 and abs(r['delta']) > 0.33) else 'FAIL'}")


if __name__ == "__main__":
    main()
