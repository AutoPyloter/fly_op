"""Slope stability, CONTINUOUS closed-loop version -- fills the last
empty cell of PROTOCOL_V2_TASK_SPECTRUM.md's 2x2 (uzamsal-YAPI x
kapali-dongu) matrix cheaply. Every other slope-stability pilot so far
(fly_slope_stability_multiseed.py) was a SINGLE-SHOT ray-to-direction
map -- RateBrain.forward() zeroes its state on every call, so the
brain never remembers earlier probes. This version keeps persistent
brain state across an entire multi-step refinement episode (like
cartpole_control.py's rollout, or malecns's continuous driving),
re-probing the FS landscape from wherever the trial circle currently is
each step -- a real iterative search, not one lookup.

Loss = mean FS achieved across the episode (lower is better, minimizing
the factor of safety = finding the critical slip surface) -- directly
optimizable end-to-end since factor_of_safety and the rays are built
from smooth (if not analytically differentiable through the search
itself) numpy ops; gradients flow through the BRAIN's recurrence and
the trajectory bookkeeping the same way cartpole's do, treating each
step's FS as a value to minimize via REINFORCE-free direct
backprop-through-trajectory is not exact for the physically-computed FS
(numpy, not torch) -- so training here uses the same imitation-style
ray/teacher-direction supervision as the single-shot version, but
APPLIED REPEATEDLY across a persistent-state multi-step episode, with
the final evaluation metric being the best FS actually reached along
the trajectory (a genuine multi-step search outcome), not one-shot
prediction error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from flyopt.benchmarks_geo import factor_of_safety, random_geometry, sample_valid_point
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta


@dataclass(frozen=True)
class SlopeContinuousConfig:
    dim: int = 3
    T_inner: int = 4
    episode_steps: int = 20
    ray_radius: float = 1.0
    step_scale: float = 1.0
    reset_state_each_step: bool = False  # ablation (2026-09-21, user question): if True, h is
    # zeroed at the start of every step instead of persisting across the episode -- isolates
    # whether the continuous variant's negative result (EXPERIMENTS.md, delta=-0.156) comes from
    # PERSISTENT brain state specifically, or from the multi-step search process itself (x moving
    # step by step) regardless of memory.


class SlopeContinuousBrain(nn.Module):
    """Same Dale's-law parameterisation as rate_brain.RateBrain /
    cartpole_control.CartpoleBrain; `rollout` keeps brain state h
    persistent across `episode_steps` moves, re-encoding fresh rays from
    the CURRENT trial-circle position each step (numpy physics/geometry,
    detached from autograd -- only the brain's own weights are trained,
    via a per-step imitation loss against the local teacher direction,
    summed over the whole episode)."""

    def __init__(self, sub_weights_csr: sparse.csr_matrix, encode_idx: np.ndarray, decode_idx: np.ndarray,
                 config: SlopeContinuousConfig, seed: int):
        super().__init__()
        self.config = config
        self.n = sub_weights_csr.shape[0]
        coo = sub_weights_csr.tocoo()
        self.register_buffer("row", torch.as_tensor(coo.row, dtype=torch.int64))
        self.register_buffer("col", torch.as_tensor(coo.col, dtype=torch.int64))
        self.register_buffer("sign", torch.as_tensor(np.sign(coo.data), dtype=torch.float32))
        self.register_buffer("encode_idx", torch.as_tensor(encode_idx, dtype=torch.int64))
        self.register_buffer("decode_idx", torch.as_tensor(decode_idx, dtype=torch.int64))

        magnitude = torch.as_tensor(np.abs(coo.data), dtype=torch.float32).clamp(min=1e-4)
        init_gain = magnitude + torch.log(-torch.expm1(-magnitude))
        self.gain = nn.Parameter(init_gain)

        gen = torch.Generator().manual_seed(seed)
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.leak_logit = nn.Parameter(torch.zeros(self.n))
        self.readout_w = nn.Parameter(torch.randn(config.dim, len(decode_idx), generator=gen) / np.sqrt(len(decode_idx)))

    def effective_weights(self) -> torch.Tensor:
        return self.sign * F.softplus(self.gain)

    def _substep(self, h, w, a, bias, inject):
        src = h.index_select(0, self.col)
        pre = torch.zeros_like(h).index_add_(0, self.row, src * w.unsqueeze(1))
        pre = pre + inject
        return (1 - a) * h + a * torch.tanh(pre + bias)

    def run_episode(self, geometry, x0: np.ndarray, train_mode: bool):
        """One multi-step refinement episode from starting point x0 (numpy,
        shape (dim,)). Returns (loss tensor or None, best_fs float,
        fs_trace list). Batch size 1 -- geometry/FS evaluation is numpy,
        not batched; cheap enough (episode_steps ~20) to run per-instance."""
        cfg = self.config
        device = self.sign.device
        w = self.effective_weights()
        a = torch.sigmoid(self.leak_logit).unsqueeze(1)
        bias = self.bias.unsqueeze(1)
        h = torch.zeros(self.n, 1, device=device)

        x = x0.copy()
        lo, hi = geometry.param_bounds()
        fx = factor_of_safety(x, geometry)
        best_fs = fx
        fs_trace = [fx]
        losses = []

        for _ in range(cfg.episode_steps):
            if cfg.reset_state_each_step:
                h = torch.zeros(self.n, 1, device=device)
            rays = _rays(x, lambda p: factor_of_safety(p, geometry), fx, cfg.ray_radius)
            target = _teacher_delta(x, rays, cfg.ray_radius)
            norm = np.linalg.norm(target)
            target_t = torch.as_tensor((target / norm if norm > 1e-8 else np.zeros(cfg.dim)), dtype=torch.float32, device=device)

            inject = torch.zeros(self.n, 1, device=device)
            inject.index_add_(0, self.encode_idx, torch.as_tensor(rays, dtype=torch.float32, device=device).unsqueeze(1))
            for _ in range(cfg.T_inner):
                h = self._substep(h, w, a, bias, inject)
            decode_h = h.index_select(0, self.decode_idx).squeeze(1)
            pred = self.readout_w @ decode_h

            if train_mode:
                losses.append(F.mse_loss(pred, target_t))

            step = pred.detach().cpu().numpy() * cfg.step_scale
            x = np.clip(x + step, lo, hi)
            fx = factor_of_safety(x, geometry)
            best_fs = min(best_fs, fx)
            fs_trace.append(fx)

        loss = torch.stack(losses).mean() if losses else None
        return loss, best_fs, fs_trace


def train_slope_continuous(brain: SlopeContinuousBrain, n_problems: int, n_episodes_per_problem: int,
                            epochs: int, lr: float, seed: int, verbose_every: int = 0,
                            geometries_per_step: int = 6) -> list[float]:
    """2026-09-18 fix: earlier version cycled ONE fixed geometry per epoch
    (round-robin through a small pool) -- each gradient step only ever
    saw a single problem instance's landscape, so consecutive epochs on
    DIFFERENT geometries pulled the readout in conflicting directions
    (verified: mean_best_fs_this_batch oscillated 2..40 across epochs,
    never settling -- EXPERIMENTS.md 2026-09-18). Fix: sample
    `geometries_per_step` FRESH random geometries every single gradient
    step (matching how the single-shot slope-stability pilot, and
    cartpole's batch=16, both average over many instances per step)."""
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    curve = []
    for epoch in range(epochs):
        opt.zero_grad()
        total_loss = []
        best_fs_this_epoch = []
        for _ in range(geometries_per_step):
            geo = random_geometry(rng)
            lo, hi = geo.param_bounds()
            for _ in range(n_episodes_per_problem):
                x0 = sample_valid_point(geo, rng)
                loss, best_fs, _ = brain.run_episode(geo, x0, train_mode=True)
                total_loss.append(loss)
                best_fs_this_epoch.append(best_fs)
        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}/{epochs}  loss={float(torch.stack(total_loss).mean()):.4f}  "
                  f"mean_best_fs_this_batch={np.mean(best_fs_this_epoch):.3f}")
        loss = torch.stack(total_loss).mean()
        loss.backward()
        opt.step()
        curve.append(float(loss.item()))
    return curve


def evaluate_slope_continuous(brain: SlopeContinuousBrain, n_problems: int, n_episodes_per_problem: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    best_fs_list = []
    with torch.no_grad():
        for _ in range(n_problems):
            geo = random_geometry(rng)
            lo, hi = geo.param_bounds()
            for _ in range(n_episodes_per_problem):
                x0 = sample_valid_point(geo, rng)
                _, best_fs, _ = brain.run_episode(geo, x0, train_mode=False)
                best_fs_list.append(best_fs)
    return float(np.mean(best_fs_list))
