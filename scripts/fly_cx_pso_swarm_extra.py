"""Extra 8 seeds (8-15) for fly_cx_pso_swarm_multiseed.py's classic-gbest
config only -- the one config whose |Cliff's delta| exceeded 0.33 in TWO
independent runs (this script's own first 8 seeds, delta=-0.406; and the
gbest arm of fly_swarm_topologies_multiseed.py using a different
subgraph-selection seed, delta=-0.375) -- both null-favoring, both
underpowered at n=8. Per the session's established rule ("n=8
threshold-passing result should not be trusted without extending to
n=16"), extending here. Same encode/CX subgraph (seed=17000) as the
original run so seeds 0-7 and 8-15 are directly poolable.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse, stats

from flyopt.benchmarks import RASTRIGIN_BOUNDS, rastrigin
from flyopt.substrates.graph_builders import er_null
from flyopt.variants.rate_brain import build_subgraph_bfs, select_connected_encode_decode
from flyopt.variants.cx_modulated_search import CXModulatedBrain, CXModulatedConfig, train_cx_modulated
from fly_cx_pso_swarm_multiseed import run_swarm

DATA_PROCESSED = "C:/projeler/fly_op/data/processed"
CELLTYPES = f"{DATA_PROCESSED}/celltypes"
DIM = 2
N_RAYS = 4
N_DECODE = 20
N_CX = 15
SUBGRAPH_SIZE = 2000
N_ITERS = 60
SEEDS = list(range(8, 16))
TRAIN_EPOCHS = 100


def main():
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/afferent_indices.npy")
    cx_pool = np.load(f"{CELLTYPES}/central_complex_indices.npy")

    encode_full, cx_verified = select_connected_encode_decode(
        base_weights, afferent, cx_pool, n_encode=N_RAYS, n_decode=N_DECODE + N_CX,
        max_hops=6, n_encode_candidates=200, seed=17000,
    )
    decode_full = cx_verified[:N_DECODE]
    cx_full = cx_verified[N_DECODE:N_DECODE + N_CX]
    all_targets = np.concatenate([decode_full, cx_full])
    sub_real, encode_idx, mapped_targets, _nodes = build_subgraph_bfs(base_weights, encode_full, all_targets, SUBGRAPH_SIZE, seed=17000)
    decode_idx = mapped_targets[: len(decode_full)]
    cx_idx = mapped_targets[len(decode_full):]
    print(f"real subgraph: {sub_real.shape[0]} nodes, {sub_real.nnz} edges")

    cfg = CXModulatedConfig(dim=DIM, T_inner=4, episode_steps=N_ITERS, ray_radius=0.3)
    train_cfg = CXModulatedConfig(dim=DIM, T_inner=4, episode_steps=20, ray_radius=0.3)
    log_path = "C:/projeler/fly_op/results/fly_cx_pso_swarm_trained_multiseed_extra.jsonl"
    results = {"real_connectome": [], "er_null": []}

    with open(log_path, "w", encoding="utf-8") as f:
        for seed in SEEDS:
            sub_null = er_null(sub_real, seed=seed)
            for name, sub in [("real_connectome", sub_real), ("er_null", sub_null)]:
                template = CXModulatedBrain(sub, encode_idx, decode_idx, cx_idx, train_cfg, seed=seed)
                train_cx_modulated(template, epochs=TRAIN_EPOCHS, lr=5e-3, seed=seed, geometries_per_step=6)
                trained_state = {
                    "readout_dir": template.readout_dir.data.clone(),
                    "readout_gain": template.readout_gain.data.clone(),
                    "readout_gain_bias": template.readout_gain_bias.data.clone(),
                    "gain": template.gain.data.clone(),
                    "bias": template.bias.data.clone(),
                    "leak_logit": template.leak_logit.data.clone(),
                }
                gbest_fx = run_swarm(sub, encode_idx, decode_idx, cx_idx, cfg, seed, trained_state=trained_state)
                results[name].append(gbest_fx)
                f.write(json.dumps({"seed": seed, "substrate": name, "gbest_fx": gbest_fx}) + "\n")
                f.flush()
                print(f"seed={seed} [{name:16s}] gbest_fx={gbest_fx:.4f}")

    real = np.array(results["real_connectome"])
    null = np.array(results["er_null"])
    print("\n=== SONUC (extra, seeds 8-15) ===")
    print(f"real: {real.tolist()}")
    print(f"null: {null.tolist()}")

    with open("C:/projeler/fly_op/results/fly_cx_pso_swarm_trained_summary_extra.json", "w", encoding="utf-8") as f:
        json.dump({"real": real.tolist(), "null": null.tolist()}, f, indent=2)

    # combine with the original 8 seeds for the pooled n=16 test
    with open("C:/projeler/fly_op/results/fly_cx_pso_swarm_trained_summary.json") as f:
        orig = json.load(f)
    combined_real = np.array(orig["real"] + real.tolist())
    combined_null = np.array(orig["null"] + null.tolist())
    wstat, pvalue = stats.wilcoxon(combined_real, combined_null)
    gt = np.sum(combined_real[:, None] < combined_null[None, :])
    lt = np.sum(combined_real[:, None] > combined_null[None, :])
    cliffs_delta = float((gt - lt) / (len(combined_real) * len(combined_null)))
    print("\n=== BIRLESTIRILMIS SONUC (n=16 tohum) ===")
    print(f"real median={np.median(combined_real):.4f}  null median={np.median(combined_null):.4f}")
    print(f"Wilcoxon p={pvalue:.5f}  Cliff's delta={cliffs_delta:.3f}")
    print(f"gate: {'PASS' if (pvalue < 0.05 and abs(cliffs_delta) > 0.33) else 'FAIL'}")

    with open("C:/projeler/fly_op/results/fly_cx_pso_swarm_trained_summary_n16.json", "w", encoding="utf-8") as f:
        json.dump({"real": combined_real.tolist(), "null": combined_null.tolist(),
                    "wilcoxon_p": float(pvalue), "cliffs_delta": cliffs_delta}, f, indent=2)
    print("wrote results/fly_cx_pso_swarm_trained_summary_n16.json")


if __name__ == "__main__":
    main()
