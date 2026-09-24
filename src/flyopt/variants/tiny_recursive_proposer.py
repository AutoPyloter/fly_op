"""Tiny-Recursive-Model-inspired proposer (informal exploration, not a
pre-registered phase -- see scripts/fly_tiny_recursive_demo.py and
EXPERIMENTS.md 2026-09-17 "HRM/TRM yerine dene").

Not a literal HRM/TRM reimplementation (those target grid/puzzle domains
like ARC-AGI/Sudoku); this adapts the core idea -- a small network that
recursively refines a latent "reasoning" state `z` and a current "answer"
`y` over several inner steps with SHARED weights (depth from recursion,
not from stacking layers), trained end-to-end by backprop -- to this
project's encode/decode contract, so it's a drop-in Proposer swap for the
connectome substrate: same landscape-ray input as FlyProposerScene, same
Substrate/Proposer-shaped usage, but a fully-trainable dense network
instead of a fixed spiking connectome + linear readout.

Per the TRM paper's own finding ("less is more"), one small shared-weight
MLP core recursed several times outperformed the original HRM's two
separate hierarchical modules -- so this uses the simpler single-core
design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from flyopt.substrate import SearchContext
from flyopt.variants.fly_proposer_scene import _rays


@dataclass(frozen=True)
class TinyRecursiveConfig:
    dim: int
    latent_dim: int = 32
    hidden_dim: int = 64
    n_recursions: int = 6
    ray_radius: float = 0.3
    decode_scale: float = 0.5


class _Core(nn.Module):
    """One shared-weight step: (z, y, input) -> (z_new, y_new). Recursed
    n_recursions times at inference AND during training (full backprop
    through the recursion -- affordable here since n_recursions is small
    and the core is tiny)."""

    def __init__(self, dim: int, n_rays: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = latent_dim + dim + n_rays
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim + dim),
        )
        self.latent_dim = latent_dim
        self.dim = dim

    def forward(self, z, y, rays):
        h = torch.cat([z, y, rays], dim=-1)
        out = self.net(h)
        dz, dy = out[..., : self.latent_dim], out[..., self.latent_dim :]
        return z + dz, y + torch.tanh(dy)  # bounded answer update, like TRM's refinement step


class TinyRecursiveProposer:
    """Proposer-protocol-shaped wrapper around `_Core`. `propose(x, ctx)`
    recurses n_recursions steps from a fresh (z=0, y=0) each call --
    stateless across calls, like FlyProposer, since each optimization step
    is treated as an independent "puzzle" (what's the best direction from
    here), not a continuing trajectory."""

    def __init__(self, config: TinyRecursiveConfig, n_rays: int, seed: int = 0):
        torch.manual_seed(seed)
        self.config = config
        self.core = _Core(config.dim, n_rays, config.latent_dim, config.hidden_dim)
        self._objective = None

    def bind_objective(self, objective) -> None:
        self._objective = objective

    def _recurse(self, rays_t: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = rays_t.shape[0] if rays_t.dim() == 2 else 1
        rays_t = rays_t.reshape(batch, -1)
        z = torch.zeros(batch, cfg.latent_dim)
        y = torch.zeros(batch, cfg.dim)
        for _ in range(cfg.n_recursions):
            z, y = self.core(z, y, rays_t)
        return y

    def propose(self, x: np.ndarray, ctx: SearchContext) -> np.ndarray:
        rays = _rays(x, self._objective, ctx.best_fx, self.config.ray_radius)
        with torch.no_grad():
            y = self._recurse(torch.as_tensor(rays, dtype=torch.float32).unsqueeze(0))
        return x + self.config.decode_scale * y.squeeze(0).numpy()

    def tell(self, x: np.ndarray, fx: float, improved: bool) -> None:
        return  # frozen after training, same rationale as the other variants


def build_and_train(
    config: TinyRecursiveConfig,
    bounds: tuple[float, float],
    n_problems: int,
    n_starts: int,
    epochs: int,
    lr: float,
    seed: int,
) -> TinyRecursiveProposer:
    """Same multi-problem/multi-start supervised scheme as
    fly_swarm_social.build_training_set (many randomly-offset Rastrigin
    instances, ideal direction = towards the known true minimum), but
    trained by Adam + MSE through the recursion instead of ridge
    regression on a fixed substrate's firing rate."""
    from flyopt.benchmarks import rastrigin

    rng = np.random.default_rng(seed)
    lo, hi = bounds
    dim = config.dim
    n_rays = 2 * dim

    proposer = TinyRecursiveProposer(config, n_rays=n_rays, seed=seed)
    opt = torch.optim.Adam(proposer.core.parameters(), lr=lr)

    X, Y = [], []
    for _ in range(n_problems):
        offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
        for _ in range(n_starts):
            x = rng.uniform(lo, hi, size=dim)
            fx = rastrigin(x - offset)
            rays = _rays(x, lambda p, off=offset: rastrigin(p - off), fx, config.ray_radius)
            target = offset - x
            norm = np.linalg.norm(target)
            target = target / norm if norm > 1e-8 else np.zeros(dim)
            X.append(rays)
            Y.append(target)

    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Y_t = torch.as_tensor(np.asarray(Y), dtype=torch.float32)

    for epoch in range(epochs):
        opt.zero_grad()
        pred = proposer._recurse(X_t)
        loss = ((pred - Y_t) ** 2).mean()
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"[tiny-recursive train] epoch {epoch:4d}/{epochs}  mse={loss.item():.4f}")

    return proposer
