"""Fly-Proposer-Scene variant (Faz S1, preregistration/faz_s1.md).

Reframes the search as a sensorimotor "scene" instead of Faz 1/1b/1c's
one-shot stateless injection, matching what a car-driving connectome demo
(fly_demos/malecns) showed works well: (1) a locally-sampled, direction-
structured percept instead of raw x injected as current, (2) persistent
substrate state across the whole search run instead of a fresh reset every
propose() call, (3) a readout trained once by imitating a cheap scripted
teacher instead of left frozen-random.

Encoding ("rays"): the objective is finite-difference-probed along +/- each
axis at radius `ray_radius`, giving 2*dim "ray" values r_k = f(x + step_k) -
f(x) (lower/negative = better direction). These are injected as constant
current into `dim*2` encode neurons for T steps, same as Faz 1b's afferent
injection but now direction-structured instead of raw-x.

Decoding: unchanged from FlyProposer — mean firing rate of the decode pool,
linear readout to Δx. What changes is *how the readout is set*: instead of
staying frozen-random for the whole run (Faz 1/1b) or being nudged online
by reward-modulated Hebbian plasticity (Faz 1c), it is fit once up front by
ridge regression against a cheap negative-gradient teacher built from the
same ray probes (`calibrate_readout`), then frozen for the search itself —
avoiding Faz 1c's small-budget online-tuning artifact (EXPERIMENTS.md
2026-09-15) while still giving every substrate a *trained*, not random,
interpreter. Ray probes and calibration steps are cheap synthetic-objective
evaluations, identical in number across every substrate kind, and are not
charged against the main search budget (PROTOCOL.md budget parity is about
substrate calls, not synthetic-function calls).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyopt.substrate import SearchContext, Substrate


@dataclass(frozen=True)
class SceneProposerConfig:
    dim: int
    n_readout: int = 50
    T: int = 15
    ray_radius: float = 0.3
    ray_scale: float = 1.0
    decode_scale: float = 0.5
    calib_ridge_lambda: float = 10.0


def _rays(x: np.ndarray, objective, fx: float, radius: float) -> np.ndarray:
    """2*dim finite-difference probes along +/- each axis: r_k = f(x+step)-fx."""
    dim = len(x)
    out = np.empty(2 * dim, dtype=np.float32)
    for i in range(dim):
        step = np.zeros(dim)
        step[i] = radius
        out[2 * i] = objective(x + step) - fx
        out[2 * i + 1] = objective(x - step) - fx
    return out


def _teacher_delta(x: np.ndarray, rays: np.ndarray, radius: float) -> np.ndarray:
    """Crude negative-gradient estimate from the same ray probes: move away
    from directions that made f worse, towards directions that made it better."""
    dim = len(x)
    grad = np.array([(rays[2 * i] - rays[2 * i + 1]) / (2 * radius) for i in range(dim)])
    return -grad


class FlyProposerScene:
    def __init__(
        self,
        substrate: Substrate,
        config: SceneProposerConfig,
        seed: int,
        encode_pool: np.ndarray | None = None,
        decode_pool: np.ndarray | None = None,
        reset_seed: int = 0,
    ):
        # `seed` fixes encode/decode indices + initial readout — shared
        # across substrate kinds (PROTOCOL.md 4.1). `reset_seed` seeds the
        # substrate's own initial dynamics state and is meant to vary with
        # the search run's seed, so a 30-seed sweep also varies initial
        # conditions, not just the starting x.
        self.substrate = substrate
        self.config = config
        self._reset_seed = reset_seed
        rng = np.random.default_rng(seed)
        n_rays = 2 * config.dim

        enc_pool = encode_pool if encode_pool is not None else np.arange(substrate.n_units)
        dec_pool = decode_pool if decode_pool is not None else np.arange(substrate.n_units)
        assert not set(enc_pool.tolist()) & set(dec_pool.tolist()), "encode_pool and decode_pool overlap"
        self.encode_indices = rng.choice(enc_pool, size=n_rays, replace=False)
        self.decode_indices = rng.choice(dec_pool, size=config.n_readout, replace=False)

        self.readout_W = rng.normal(0, 1.0 / np.sqrt(config.n_readout), size=(config.dim, config.n_readout))
        self._state = None  # persistent across the whole search run; reset() only on first use

    def _inject_and_read(self, rays: np.ndarray) -> np.ndarray:
        cfg = self.config
        if self._state is None:
            self._state = self.substrate.reset(seed=self._reset_seed)
        stim = np.zeros(self.substrate.n_units, dtype=np.float32)
        stim[self.encode_indices] = rays * cfg.ray_scale
        spike_accum = np.zeros(self.substrate.n_units)
        for _ in range(cfg.T):
            spikes, self._state = self.substrate.step(stim, self._state)
            spike_accum += spikes
        return spike_accum[self.decode_indices] / cfg.T

    def propose(self, x: np.ndarray, ctx: SearchContext) -> np.ndarray:
        # (1+1) elitist hillclimb invariant (search_loop.run_hillclimb): x is
        # only ever updated to an improving candidate, so ctx.best_fx == f(x)
        # exactly at every call — no need to re-evaluate the objective at x.
        rays = _rays(x, self._objective, ctx.best_fx, self.config.ray_radius)
        firing_rate = self._inject_and_read(rays)
        delta = self.readout_W @ firing_rate
        return x + self.config.decode_scale * delta

    def tell(self, x: np.ndarray, fx: float, improved: bool) -> None:
        return  # readout is frozen after calibration; see module docstring

    def bind_objective(self, objective) -> None:
        """The scene encoding needs to probe the objective locally; injected
        after construction so callers building the proposer don't need to
        thread it through every constructor (PROTOCOL.md keeps Substrate/
        Proposer minimal, this is a Scene-specific extension, not a core
        interface change)."""
        self._objective = objective


def calibrate_readout(
    proposer: FlyProposerScene,
    objective,
    dim: int,
    bounds: tuple[float, float],
    n_steps: int,
    seed: int,
) -> None:
    """One-round DAgger-0 style calibration: a scripted negative-gradient
    teacher walks `n_steps` ticks, the proposer's substrate watches (state
    persists across ticks, matching how it will run during search), and the
    readout is ridge-regressed to reproduce the teacher's move from the
    firing rate it produced. Mutates proposer.readout_W in place."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    x = rng.uniform(lo, hi, size=dim)
    fx = objective(x)

    X, Y = [], []
    for _ in range(n_steps):
        rays = _rays(x, objective, fx, proposer.config.ray_radius)
        firing_rate = proposer._inject_and_read(rays)
        target = _teacher_delta(x, rays, proposer.config.ray_radius)
        X.append(firing_rate)
        Y.append(target)
        x = np.clip(x + proposer.config.decode_scale * target, lo, hi)
        fx = objective(x)

    X = np.asarray(X)
    Y = np.asarray(Y)
    lam = proposer.config.calib_ridge_lambda
    n_readout = X.shape[1]
    proposer.readout_W = (Y.T @ X) @ np.linalg.inv(X.T @ X + lam * np.eye(n_readout))
