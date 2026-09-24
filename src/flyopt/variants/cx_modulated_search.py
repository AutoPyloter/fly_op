"""Central-complex-modulated search (2026-09-18, idea #4 from the
brainstorm: "gercek osilator/navigasyon devresini kullan"). Instead of a
hand-scheduled annealing (scene_anneal's linear fly_weight decay), a
SEPARATE readout from the fly's own central complex (CX -- the insect
brain's known internal-state/navigation-state region) produces a
per-step exploration gain, learned end-to-end from the same persistent,
multi-step rollout the network is already doing. The idea: let the
network's own internal dynamics decide when to explore vs exploit,
instead of us imposing a fixed schedule from outside.

Two decode heads on the SAME persistent recurrent state:
  - `readout_dir` (from a sensory/general decode pool) -> search direction
  - `readout_gain` (from CX neurons specifically) -> scalar in (0, 1),
    sigmoid-scaled, multiplies the applied step size each move

Task: same multi-problem Rastrigin family as every other 2026-09-18
pilot, but framed as a persistent multi-step episode (like
slope_continuous.py) so the CX gain has something temporal to modulate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from flyopt.benchmarks import rastrigin
from flyopt.variants.fly_proposer_scene import _rays, _teacher_delta


@dataclass(frozen=True)
class CXModulatedConfig:
    dim: int = 2
    T_inner: int = 4
    episode_steps: int = 20
    ray_radius: float = 0.3


class CXModulatedBrain(nn.Module):
    def __init__(self, sub_weights_csr: sparse.csr_matrix, encode_idx: np.ndarray, decode_idx: np.ndarray,
                 cx_idx: np.ndarray, config: CXModulatedConfig, seed: int):
        super().__init__()
        self.config = config
        self.n = sub_weights_csr.shape[0]
        coo = sub_weights_csr.tocoo()
        self.register_buffer("row", torch.as_tensor(coo.row, dtype=torch.int64))
        self.register_buffer("col", torch.as_tensor(coo.col, dtype=torch.int64))
        self.register_buffer("sign", torch.as_tensor(np.sign(coo.data), dtype=torch.float32))
        self.register_buffer("encode_idx", torch.as_tensor(encode_idx, dtype=torch.int64))
        self.register_buffer("decode_idx", torch.as_tensor(decode_idx, dtype=torch.int64))
        self.register_buffer("cx_idx", torch.as_tensor(cx_idx, dtype=torch.int64))

        magnitude = torch.as_tensor(np.abs(coo.data), dtype=torch.float32).clamp(min=1e-4)
        init_gain = magnitude + torch.log(-torch.expm1(-magnitude))
        self.gain = nn.Parameter(init_gain)

        gen = torch.Generator().manual_seed(seed)
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.leak_logit = nn.Parameter(torch.zeros(self.n))
        self.readout_dir = nn.Parameter(torch.randn(config.dim, len(decode_idx), generator=gen) / np.sqrt(len(decode_idx)))
        self.readout_gain = nn.Parameter(torch.randn(len(cx_idx), generator=gen) / np.sqrt(len(cx_idx)))
        self.readout_gain_bias = nn.Parameter(torch.zeros(()))

    def effective_weights(self) -> torch.Tensor:
        return self.sign * F.softplus(self.gain)

    def _substep(self, h, w, a, bias, inject):
        src = h.index_select(0, self.col)
        pre = torch.zeros_like(h).index_add_(0, self.row, src * w.unsqueeze(1))
        pre = pre + inject
        return (1 - a) * h + a * torch.tanh(pre + bias)

    def run_episode_e2e(self, f, f_torch, x0: np.ndarray, train_mode: bool):
        """End-to-end differentiable variant (2026-09-18, following the
        documented limitation in EXPERIMENTS.md: in `run_episode` below,
        `readout_gain`/`readout_gain_bias` NEVER receive gradient because
        the loss only supervises `pred_dir` against a teacher direction,
        while `gain` is applied via `.item()`/`.detach().numpy()`, fully
        outside the autograd graph -- so the docstring's "learned
        end-to-end" claim was false for the gain readout specifically.
        Here `x` stays a torch tensor across the whole rollout and the
        loss is the actual task objective (mean Rastrigin value along the
        trajectory), so gain's effect on step size -- and therefore on
        outcome -- finally has a gradient path back to `readout_gain`.
        Sensory ray-casting still uses a detached numpy snapshot of `x`
        (a legitimate non-differentiable "observation", same as before);
        only the applied step and its consequence on fx are backpropped.
        """
        cfg = self.config
        w = self.effective_weights()
        a = torch.sigmoid(self.leak_logit).unsqueeze(1)
        bias = self.bias.unsqueeze(1)
        h = torch.zeros(self.n, 1)

        x_t = torch.as_tensor(x0, dtype=torch.float32)
        x_np = x0.copy()
        fx = f(x_np)
        best_fx = fx
        fx_losses = []
        gains_seen = []

        for _ in range(cfg.episode_steps):
            rays = _rays(x_np, f, fx, cfg.ray_radius)

            inject = torch.zeros(self.n, 1)
            inject.index_add_(0, self.encode_idx, torch.as_tensor(rays, dtype=torch.float32).unsqueeze(1))
            for _ in range(cfg.T_inner):
                h = self._substep(h, w, a, bias, inject)

            decode_h = h.index_select(0, self.decode_idx).squeeze(1)
            pred_dir = self.readout_dir @ decode_h

            cx_h = h.index_select(0, self.cx_idx).squeeze(1)
            gain = torch.sigmoid(self.readout_gain @ cx_h + self.readout_gain_bias)
            gains_seen.append(float(gain.item()))

            x_t = x_t + gain * pred_dir
            fx_t = f_torch(x_t)
            if train_mode:
                fx_losses.append(fx_t)

            x_np = x_t.detach().numpy()
            fx = float(fx_t.item())
            best_fx = min(best_fx, fx)

        loss = torch.stack(fx_losses).mean() if fx_losses else None
        return loss, best_fx, gains_seen

    def run_episode(self, f, x0: np.ndarray, train_mode: bool):
        cfg = self.config
        w = self.effective_weights()
        a = torch.sigmoid(self.leak_logit).unsqueeze(1)
        bias = self.bias.unsqueeze(1)
        h = torch.zeros(self.n, 1)

        x = x0.copy()
        fx = f(x)
        best_fx = fx
        losses = []
        gains_seen = []

        for _ in range(cfg.episode_steps):
            rays = _rays(x, f, fx, cfg.ray_radius)
            target = _teacher_delta(x, rays, cfg.ray_radius)
            norm = np.linalg.norm(target)
            target_t = torch.as_tensor((target / norm if norm > 1e-8 else np.zeros(cfg.dim)), dtype=torch.float32)

            inject = torch.zeros(self.n, 1)
            inject.index_add_(0, self.encode_idx, torch.as_tensor(rays, dtype=torch.float32).unsqueeze(1))
            for _ in range(cfg.T_inner):
                h = self._substep(h, w, a, bias, inject)

            decode_h = h.index_select(0, self.decode_idx).squeeze(1)
            pred_dir = self.readout_dir @ decode_h

            cx_h = h.index_select(0, self.cx_idx).squeeze(1)
            gain = torch.sigmoid(self.readout_gain @ cx_h + self.readout_gain_bias)
            gains_seen.append(float(gain.item()))

            if train_mode:
                losses.append(F.mse_loss(pred_dir, target_t))

            step = pred_dir.detach().numpy() * float(gain.item())
            x = x + step
            fx = f(x)
            best_fx = min(best_fx, fx)

        loss = torch.stack(losses).mean() if losses else None
        return loss, best_fx, gains_seen


def train_cx_modulated(brain: CXModulatedBrain, epochs: int, lr: float, seed: int,
                        geometries_per_step: int = 6, dim: int = 2, bounds=(-5.12, 5.12),
                        grad_clip: float = 1.0) -> list[float]:
    """2026-09-18: `grad_clip` added after discovering that at
    episode_steps=60 (the swarm-training config) the 240-step recurrent
    backprop unroll (60 * T_inner=4) overflows to inf/nan gradients
    within 1-2 epochs -- same failure class as cartpole_control.py's
    exploding-gradient outlier, but severe enough here to NaN every
    trained parameter (except readout_gain/readout_gain_bias, which never
    receive gradient at all since the loss never depends on them -- see
    run_episode). At episode_steps=20 (the single-agent config) raw grad
    norms were already absurd (~1e13-1e19) even though Adam's per-parameter
    normalization happened to keep parameters finite -- clipping here
    makes that no longer a matter of luck.
    """
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    curve = []
    for epoch in range(epochs):
        opt.zero_grad()
        total_loss = []
        for _ in range(geometries_per_step):
            offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
            f = lambda p, off=offset: rastrigin(p - off)
            x0 = rng.uniform(lo, hi, size=dim)
            loss, _, _ = brain.run_episode(f, x0, train_mode=True)
            total_loss.append(loss)
        loss = torch.stack(total_loss).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), grad_clip)
        opt.step()
        curve.append(float(loss.item()))
    return curve


def evaluate_cx_modulated(brain: CXModulatedBrain, n_problems: int, seed: int, dim: int = 2, bounds=(-5.12, 5.12)) -> float:
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    best_list = []
    with torch.no_grad():
        for _ in range(n_problems):
            offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
            f = lambda p, off=offset: rastrigin(p - off)
            x0 = rng.uniform(lo, hi, size=dim)
            _, best_fx, _ = brain.run_episode(f, x0, train_mode=False)
            best_list.append(best_fx)
    return float(np.mean(best_list))


def _rastrigin_torch(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return 10 * n + torch.sum(x**2 - 10 * torch.cos(2 * np.pi * x))


def train_cx_modulated_e2e(brain: CXModulatedBrain, epochs: int, lr: float, seed: int,
                            geometries_per_step: int = 6, dim: int = 2, bounds=(-5.12, 5.12),
                            grad_clip: float = 1.0) -> list[float]:
    """Trains via run_episode_e2e -- see that method's docstring. Same
    grad-clip precaution as train_cx_modulated even though the 20-step
    (80 total with T_inner) horizon was already confirmed NaN-free;
    the loss here backprops through an additional `x` accumulation
    chain on top of the recurrent one, so clipping stays cheap insurance."""
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    curve = []
    for epoch in range(epochs):
        opt.zero_grad()
        total_loss = []
        for _ in range(geometries_per_step):
            offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
            offset_t = torch.as_tensor(offset, dtype=torch.float32)
            f = lambda p, off=offset: rastrigin(p - off)
            f_torch = lambda p, off=offset_t: _rastrigin_torch(p - off)
            x0 = rng.uniform(lo, hi, size=dim)
            loss, _, _ = brain.run_episode_e2e(f, f_torch, x0, train_mode=True)
            total_loss.append(loss)
        loss = torch.stack(total_loss).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), grad_clip)
        opt.step()
        curve.append(float(loss.item()))
    return curve


def evaluate_cx_modulated_e2e(brain: CXModulatedBrain, n_problems: int, seed: int, dim: int = 2, bounds=(-5.12, 5.12)) -> float:
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    best_list = []
    with torch.no_grad():
        for _ in range(n_problems):
            offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
            offset_t = torch.as_tensor(offset, dtype=torch.float32)
            f = lambda p, off=offset: rastrigin(p - off)
            f_torch = lambda p, off=offset_t: _rastrigin_torch(p - off)
            x0 = rng.uniform(lo, hi, size=dim)
            _, best_fx, _ = brain.run_episode_e2e(f, f_torch, x0, train_mode=False)
            best_list.append(best_fx)
    return float(np.mean(best_list))
