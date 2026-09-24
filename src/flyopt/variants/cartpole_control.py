"""Cartpole balance control -- PROTOCOL_V2_TASK_SPECTRUM.md spectrum
point #4, the "continuous closed-loop control, cheap" gap between static
function tasks (Rastrigin, slope stability, Fly-Surrogate) and malecns's
expensive full embodied driving. Classic control-theory benchmark
(Barto/Sutton/Anderson 1983 equations), chosen specifically because it
is CONTINUOUS and CLOSED-LOOP (state feeds back into the network every
physics step, unlike a static landscape) while being cheap enough
(4-D state, scalar force, ~50-step episodes) to run many seeds on CPU.

Physics and brain dynamics are BOTH smooth/differentiable, so the whole
rollout (brain recurrence coupled step-by-step with cartpole physics) is
trained end-to-end by backprop -- no scripted teacher, no imitation
target: the loss IS the control objective (keep the pole upright, the
cart centered), integrated over the episode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
POLE_HALF_LENGTH = 0.5
DT = 0.02
FORCE_SCALE = 10.0  # brain output (roughly unit-scale) -> physical force (N)


@dataclass(frozen=True)
class CartpoleConfig:
    T_inner: int = 2  # LIF-style recurrent sub-steps per physics tick (brain settles faster than the cart moves)
    episode_steps: int = 50


def cartpole_step(state: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    """state: (batch, 4) = (x, x_dot, theta, theta_dot). force: (batch,).
    Standard nonlinear cartpole dynamics, semi-implicit Euler -- every op
    here is a plain differentiable tensor op, no branching."""
    x, x_dot, theta, theta_dot = state.unbind(-1)
    costheta, sintheta = torch.cos(theta), torch.sin(theta)
    total_mass = CART_MASS + POLE_MASS
    temp = (force + POLE_MASS * POLE_HALF_LENGTH * theta_dot ** 2 * sintheta) / total_mass
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * costheta ** 2 / total_mass)
    )
    x_acc = temp - POLE_MASS * POLE_HALF_LENGTH * theta_acc * costheta / total_mass

    x_dot = x_dot + DT * x_acc
    x = x + DT * x_dot
    theta_dot = theta_dot + DT * theta_acc
    theta = theta + DT * theta_dot
    return torch.stack([x, x_dot, theta, theta_dot], dim=-1)


class CartpoleBrain(nn.Module):
    """Same connectome parameterisation as rate_brain.RateBrain (Dale's
    law fixed sign, trainable magnitude via softplus, trainable
    bias/leak), but `forward` runs a full closed-loop episode: brain
    state and cart state co-evolve, coupled every physics tick."""

    def __init__(self, sub_weights_csr: sparse.csr_matrix, encode_idx: np.ndarray, decode_idx: np.ndarray,
                 config: CartpoleConfig, seed: int):
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
        self.readout_w = nn.Parameter(torch.randn(len(decode_idx), generator=gen) / np.sqrt(len(decode_idx)))
        self.readout_b = nn.Parameter(torch.zeros(()))
        self.input_scale = nn.Parameter(torch.ones(4))  # per-channel scale: x/x_dot/theta/theta_dot ranges differ

    def effective_weights(self) -> torch.Tensor:
        return self.sign * F.softplus(self.gain)

    def _brain_substep(self, h: torch.Tensor, w: torch.Tensor, a: torch.Tensor, bias: torch.Tensor,
                        inject: torch.Tensor) -> torch.Tensor:
        src = h.index_select(0, self.col)
        messages = src * w.unsqueeze(1)
        pre = torch.zeros_like(h).index_add_(0, self.row, messages)
        pre = pre + inject
        return (1 - a) * h + a * torch.tanh(pre + bias)

    def rollout(self, init_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """init_state: (batch, 4). Returns (cost, trajectory (batch, episode_steps, 4))."""
        cfg = self.config
        batch = init_state.shape[0]
        w = self.effective_weights()
        a = torch.sigmoid(self.leak_logit).unsqueeze(1)
        bias = self.bias.unsqueeze(1)
        h = torch.zeros(self.n, batch)
        state = init_state
        total_cost = torch.zeros(batch)
        traj = []

        for _ in range(cfg.episode_steps):
            scaled = state * self.input_scale
            inject = torch.zeros(self.n, batch).index_add_(0, self.encode_idx, scaled.t())
            for _ in range(cfg.T_inner):
                h = self._brain_substep(h, w, a, bias, inject)
            force = (h.index_select(0, self.decode_idx).t() @ self.readout_w + self.readout_b) * FORCE_SCALE
            state = cartpole_step(state, force)
            total_cost = total_cost + state[:, 2] ** 2 + 0.1 * state[:, 0] ** 2  # theta^2 + 0.1*x^2
            traj.append(state)

        return total_cost.mean() / cfg.episode_steps, torch.stack(traj, dim=1)


def sample_init_states(batch: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, size=batch)
    x_dot = rng.uniform(-0.5, 0.5, size=batch)
    theta = rng.uniform(-0.2, 0.2, size=batch)  # near-upright, ~11 degrees max
    theta_dot = rng.uniform(-0.5, 0.5, size=batch)
    return torch.as_tensor(np.stack([x, x_dot, theta, theta_dot], axis=1), dtype=torch.float32)


def train_cartpole(brain: CartpoleBrain, epochs: int, lr: float, batch: int, seed: int,
                    grad_clip: float = 1.0) -> list[float]:
    """2026-09-18: `grad_clip` added after the first 8-seed run showed one
    wild outlier (seed=3 real_connectome eval_cost=15.9 vs 1.7-5.4 for
    everything else, EXPERIMENTS.md) -- suspected exploding gradients
    through the 50-step coupled brain+physics backward pass for that
    seed's particular real-connectome initialization. Standard fix,
    testing whether it removes the outlier and changes the aggregate
    result."""
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    curve = []
    for epoch in range(epochs):
        init_state = sample_init_states(batch, seed=seed * 10000 + epoch)
        opt.zero_grad()
        cost, _ = brain.rollout(init_state)
        cost.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), grad_clip)
        opt.step()
        curve.append(float(cost.item()))
    return curve
