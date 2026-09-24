"""Small-scale Fly-RL probe (PROTOCOL.md 2.5: "plastisite acik, gercek
sinaptik agirliklar ogreniyor"). Never run before this -- every prior
variant (FlyProposer/Scene/SwarmSocial) only ever trained the readout,
explicitly leaving the connectome's own synapses untouched (see those
files' docstrings). This is a deliberately small, cheap first look,
requested 2026-09-17 ("kucuk olcekte dene, birkac yuz sinapsla basla").

The LIF engine (sim/lif_vectorized.py) converts spikes to numpy every
step and uses a hard threshold -- not built for backprop. Rather than
re-engineer it for surrogate-gradient autograd (a much bigger project),
this optimizes a few hundred real synapse weights with the SAME
zeroth-order (1+1) hill-climb search the whole project already uses
everywhere else (experiment/search_loop.py) -- perturb, re-simulate,
keep if it reduces the readout's prediction error on a held-out batch.

The readout is held FIXED during this (already calibrated by
fly_proposer_scene.calibrate_readout) so the experiment isolates one
question: does nudging real synapses, with a good interpreter already in
place, help beyond what the interpreter alone gets? -- the same
isolation logic as Faz 1c (EXPERIMENTS.md 2026-09-15).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from flyopt.variants.fly_proposer_scene import FlyProposerScene, _rays


def select_synapses(weights_csr: sparse.csr_matrix, n_synapses: int, seed: int) -> np.ndarray:
    """Row indices into the CSR's underlying COO edge list -- NOT neuron
    indices. Picked uniformly at random from all real, nonzero edges.

    First attempt (2026-09-17): found nothing in 150 iterations. Superseded
    by `select_targeted_synapses` below -- kept for reference/comparison."""
    rng = np.random.default_rng(seed)
    n_edges = weights_csr.nnz
    return rng.choice(n_edges, size=n_synapses, replace=False)


def select_targeted_synapses(
    weights_csr: sparse.csr_matrix,
    encode_indices: np.ndarray,
    decode_indices: np.ndarray,
    n_synapses: int,
    seed: int,
) -> np.ndarray:
    """Synapses one hop from where the signal actually is: edges INTO the
    exact decode neurons (directly shape what the readout sees) and edges
    OUT of the exact encode neurons (directly driven by the injected
    stimulus). Far smaller, causally-relevant pool than the whole 15M-edge
    graph -- 2026-09-17 follow-up ("biraz daha detayli ara") after random
    selection found zero improving perturbations in 150 iterations."""
    rng = np.random.default_rng(seed)
    coo = weights_csr.tocoo()
    touches_decode = np.isin(coo.row, decode_indices)
    touches_encode = np.isin(coo.col, encode_indices)
    pool = np.flatnonzero(touches_decode | touches_encode)
    n = min(n_synapses, len(pool))
    return rng.choice(pool, size=n, replace=False)


def perturbed_weights(base_coo, synapse_idx: np.ndarray, theta: np.ndarray) -> sparse.csr_matrix:
    """base_coo's .data at the selected positions gets `theta` ADDED (not
    replaced) -- theta=0 reproduces the exact original connectome."""
    data = base_coo.data.copy()
    data[synapse_idx] = data[synapse_idx] + theta
    return sparse.coo_matrix((data, (base_coo.row, base_coo.col)), shape=base_coo.shape).tocsr()


def build_eval_batch(dim: int, bounds: tuple[float, float], n_problems: int, n_starts: int, seed: int):
    """Same multi-problem/multi-start scheme as fly_swarm_social.py's
    training set, held out here purely as an evaluation batch (ideal
    direction towards each instance's known true minimum)."""
    from flyopt.benchmarks import rastrigin

    rng = np.random.default_rng(seed)
    lo, hi = bounds
    samples = []
    for _ in range(n_problems):
        offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
        f = lambda x, off=offset: rastrigin(x - off)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi, size=dim)
            fx = f(x)
            target = offset - x
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(dim)
            samples.append((x, fx, f, target))
    return samples


def evaluate(weights_csr, proposer_template: FlyProposerScene, batch) -> float:
    """Mean squared error between the (fixed, already-calibrated) readout's
    predicted direction and each sample's ideal direction, run on
    `weights_csr` (base or perturbed). Rebuilds a fresh substrate each
    call -- correct but not free (~0.5-1s for FlyWire's 15M edges), the
    dominant per-fitness-evaluation cost here."""
    substrate = SparseRecurrentSubstrate(weights_csr)
    cfg = proposer_template.config
    errs = []
    for x, fx, f, target in batch:
        rays = _rays(x, f, fx, cfg.ray_radius) * cfg.ray_scale
        stim = np.zeros(substrate.n_units, dtype=np.float32)
        stim[proposer_template.encode_indices] = rays
        state = substrate.reset(seed=0)
        spike_accum = np.zeros(substrate.n_units)
        for _ in range(cfg.T):
            spikes, state = substrate.step(stim, state)
            spike_accum += spikes
        firing_rate = spike_accum[proposer_template.decode_indices] / cfg.T
        pred = proposer_template.readout_W @ firing_rate
        errs.append(float(np.sum((pred - target) ** 2)))
    return float(np.mean(errs))


def hillclimb_synapses(
    base_weights_csr,
    proposer_template: FlyProposerScene,
    synapse_idx: np.ndarray,
    budget: int,
    sigma: float,
    batch,
    seed: int,
    block_frac: float = 0.15,
    adapt_window: int = 15,
):
    """(1+1) elitist hill-climb over the given synapse weights, upgraded
    (2026-09-17, "biraz daha detayli ara") past the first naive attempt
    (EXPERIMENTS.md 2026-09-17, 150 iters, zero improvements found) in two
    ways known to matter for high-dimensional zeroth-order search:

    1. Rechenberg's 1/5 success rule: every `adapt_window` iterations,
       sigma *= 1.22 if the success rate over that window was > 1/5, else
       *= 0.82 -- classic ES step-size adaptation, no gradient needed.
    2. Block-coordinate perturbation: each iteration only perturbs a random
       `block_frac` fraction of the synapses (the rest of theta unchanged)
       instead of all of them at once -- a full-dimensional isotropic step
       is much less likely to land on an improving direction as dimension
       grows; perturbing a small block at a time explores more like
       coordinate descent while still allowing correlated multi-synapse
       moves within each block.
    """
    rng = np.random.default_rng(seed)
    base_coo = base_weights_csr.tocoo()
    n_synapses = len(synapse_idx)
    typical_mag = float(np.median(np.abs(base_coo.data[synapse_idx])))
    block_size = max(1, int(round(n_synapses * block_frac)))

    theta = np.zeros(n_synapses)
    best_err = evaluate(base_weights_csr, proposer_template, batch)
    curve = [best_err]
    print(f"[synapse hillclimb v2] baseline mse={best_err:.4f}  typical_mag={typical_mag:.2f}  "
          f"n_synapses={n_synapses}  block_size={block_size}")

    successes_in_window = 0
    for it in range(budget):
        block = rng.choice(n_synapses, size=block_size, replace=False)
        candidate = theta.copy()
        candidate[block] = candidate[block] + rng.normal(0, sigma * typical_mag, size=block_size)
        w = perturbed_weights(base_coo, synapse_idx, candidate)
        err = evaluate(w, proposer_template, batch)
        improved = err < best_err
        if improved:
            theta, best_err = candidate, err
            successes_in_window += 1
            print(f"[synapse hillclimb v2] iter {it:4d}  mse={best_err:.4f}  sigma={sigma:.4f}  (improved)")
        curve.append(best_err)

        if (it + 1) % adapt_window == 0:
            rate = successes_in_window / adapt_window
            sigma = sigma * 1.22 if rate > 0.2 else sigma * 0.82
            successes_in_window = 0
            print(f"[synapse hillclimb v2] iter {it:4d}  success_rate={rate:.2f}  new sigma={sigma:.4f}")

    return theta, synapse_idx, curve
