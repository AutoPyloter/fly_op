"""Fly-Swarm-Social proposer (informal exploration, not a pre-registered
phase -- see scripts/fly_swarm_social_demo.py and EXPERIMENTS.md
2026-09-17 "son bir deneme").

Two things this variant adds over fly_proposer_scene.FlyProposerScene:

1. **Multi-problem training** instead of one scripted-teacher rollout on a
   single landscape: many randomly-offset Rastrigin instances, many random
   start points each, ideal direction = towards that instance's *known*
   true minimum (cheap to compute exactly because we constructed the
   offset). Much richer and more diverse supervision than Faz S1's
   single-rollout calibration.

2. **Social rays**: alongside the landscape gradient rays (`_rays` in
   fly_proposer_scene.py), each fly also perceives other flies angularly --
   bin the search-space plane into `n_social_rays` angular sectors around
   the fly's own position; in each sector, the strongest neighbor signal is
   quality/(1+distance), quality = -fitness normalized so a better
   (lower-fitness) neighbor is "taller". This is the same construction as
   the landscape rays (a fixed-size panoramic sample), just of neighbors
   instead of terrain.

Calibration target blends both: ideal_dir = normalize((true_min - x) +
social_teacher_weight * (best_phantom_neighbor - x)) -- teaches the
readout to combine "go downhill" and "go toward good neighbors" from one
ridge regression, instead of hand-coding PSO's social term.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyopt.substrate import SearchContext, Substrate
from flyopt.variants.fly_proposer_scene import _rays


@dataclass(frozen=True)
class SwarmSocialConfig:
    dim: int
    n_readout: int = 50
    T: int = 15
    ray_radius: float = 0.3
    ray_scale: float = 1.0
    n_social_rays: int = 8
    social_scale: float = 1.0
    decode_scale: float = 0.5
    social_teacher_weight: float = 0.5
    calib_ridge_lambda: float = 10.0


def _social_rays(x: np.ndarray, neighbor_x: np.ndarray, neighbor_fx: np.ndarray, n_sectors: int) -> np.ndarray:
    """Angular-binned panorama of other flies: sector k's value is the
    strongest quality/(1+distance) among neighbors falling in that sector,
    0 if none. `x` is 2-D (dim must be 2 for "angle" to mean anything)."""
    out = np.zeros(n_sectors, dtype=np.float32)
    if len(neighbor_x) == 0:
        return out
    rel = neighbor_x - x
    dist = np.linalg.norm(rel, axis=1)
    angle = np.arctan2(rel[:, 1], rel[:, 0])  # [-pi, pi]
    sector = ((angle + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    quality = -neighbor_fx  # lower fitness (better) -> higher quality
    quality = quality - quality.min() + 1e-6  # keep positive
    signal = quality / (1.0 + dist)
    for k in range(n_sectors):
        mask = sector == k
        if mask.any():
            out[k] = signal[mask].max()
    return out


class FlySwarmSocial:
    def __init__(
        self,
        substrate: Substrate,
        config: SwarmSocialConfig,
        seed: int,
        encode_pool: np.ndarray | None = None,
        decode_pool: np.ndarray | None = None,
        reset_seed: int = 0,
    ):
        self.substrate = substrate
        self.config = config
        self._reset_seed = reset_seed
        n_inputs = 2 * config.dim + config.n_social_rays
        rng = np.random.default_rng(seed)

        enc_pool = encode_pool if encode_pool is not None else np.arange(substrate.n_units)
        dec_pool = decode_pool if decode_pool is not None else np.arange(substrate.n_units)
        assert not set(enc_pool.tolist()) & set(dec_pool.tolist()), "encode_pool and decode_pool overlap"
        self.encode_indices = rng.choice(enc_pool, size=n_inputs, replace=False)
        self.decode_indices = rng.choice(dec_pool, size=config.n_readout, replace=False)

        self.readout_W = rng.normal(0, 1.0 / np.sqrt(config.n_readout), size=(config.dim, config.n_readout))
        self._state = None
        self._neighbor_x = np.empty((0, config.dim))
        self._neighbor_fx = np.empty(0)
        self._objective = None

    def bind_objective(self, objective) -> None:
        self._objective = objective

    def set_neighbors(self, neighbor_x: np.ndarray, neighbor_fx: np.ndarray) -> None:
        """Called by the swarm loop each step with every *other* fly's
        current position/fitness (this fly excluded)."""
        self._neighbor_x = neighbor_x
        self._neighbor_fx = neighbor_fx

    def _encode_vector(self, x: np.ndarray, fx: float) -> np.ndarray:
        cfg = self.config
        land = _rays(x, self._objective, fx, cfg.ray_radius) * cfg.ray_scale
        social = _social_rays(x, self._neighbor_x, self._neighbor_fx, cfg.n_social_rays) * cfg.social_scale
        return np.concatenate([land, social])

    def _inject_and_read(self, encoded: np.ndarray) -> np.ndarray:
        cfg = self.config
        if self._state is None:
            self._state = self.substrate.reset(seed=self._reset_seed)
        stim = np.zeros(self.substrate.n_units, dtype=np.float32)
        stim[self.encode_indices] = encoded
        spike_accum = np.zeros(self.substrate.n_units)
        for _ in range(cfg.T):
            spikes, self._state = self.substrate.step(stim, self._state)
            spike_accum += spikes
        return spike_accum[self.decode_indices] / cfg.T

    def propose(self, x: np.ndarray, ctx: SearchContext) -> np.ndarray:
        encoded = self._encode_vector(x, ctx.best_fx)
        firing_rate = self._inject_and_read(encoded)
        delta = self.readout_W @ firing_rate
        return x + self.config.decode_scale * delta

    def tell(self, x: np.ndarray, fx: float, improved: bool) -> None:
        return  # frozen after calibration, same rationale as FlyProposerScene


def build_training_set(
    proposer: FlySwarmSocial,
    dim: int,
    bounds: tuple[float, float],
    n_problems: int,
    n_starts: int,
    n_phantom_neighbors: int,
    seed: int,
):
    """Many randomly-offset Rastrigin instances x many random starts, each
    with a few random 'phantom' neighbor flies (for the social-ray input
    and the social pull in the target). Returns (X firing rates, Y ideal
    directions) ready for ridge regression."""
    from flyopt.benchmarks import rastrigin

    rng = np.random.default_rng(seed)
    lo, hi = bounds
    cfg = proposer.config
    X, Y = [], []

    for _ in range(n_problems):
        offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
        f = lambda x, off=offset: rastrigin(x - off)
        proposer.bind_objective(f)  # _encode_vector's landscape rays must probe THIS instance

        for _ in range(n_starts):
            x = rng.uniform(lo, hi, size=dim)
            fx = f(x)

            n_neighbors = rng.integers(0, n_phantom_neighbors + 1)
            neighbor_x = rng.uniform(lo, hi, size=(n_neighbors, dim))
            neighbor_fx = np.array([f(nx) for nx in neighbor_x]) if n_neighbors else np.empty(0)
            proposer.set_neighbors(neighbor_x, neighbor_fx)

            encoded = proposer._encode_vector(x, fx)
            firing_rate = proposer._inject_and_read(encoded)

            to_min = offset - x
            target = to_min.copy()
            if n_neighbors:
                best_j = int(np.argmin(neighbor_fx))
                if neighbor_fx[best_j] < fx:  # only follow a neighbor that's actually better
                    to_neighbor = neighbor_x[best_j] - x
                    target = to_min + cfg.social_teacher_weight * to_neighbor
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(dim)

            X.append(firing_rate)
            Y.append(target)

    return np.asarray(X), np.asarray(Y)


def calibrate_from_dataset(proposer: FlySwarmSocial, X: np.ndarray, Y: np.ndarray) -> None:
    lam = proposer.config.calib_ridge_lambda
    n_readout = X.shape[1]
    proposer.readout_W = (Y.T @ X) @ np.linalg.inv(X.T @ X + lam * np.eye(n_readout))
