"""Fly-Surrogate (PROTOCOL.md #6, never tried before today -- "Pahalı
objective function'ı taklit ediyor"). Different computational role from
every proposer tried so far: instead of "propose a direction from here",
predict the objective's VALUE at a point -- regression, not navigation.
Reuses rate_brain.py's differentiable substrate + verified-connectivity
encode/decode selection unchanged; only the task changes (scalar
function-value prediction instead of direction prediction), so this file
is intentionally thin.

Encoding: raw coordinates x injected directly into `dim` encode neurons
(no finite-difference rays -- there is nothing to probe locally, the
network must learn the GLOBAL shape of f from many (x, f(x)) examples).
Readout: a single scalar (n_readout decode neurons -> 1 linear unit).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


@dataclass(frozen=True)
class SurrogateConfig:
    dim: int
    T: int = 8
    input_scale: float = 1.0


class FlySurrogate(nn.Module):
    def __init__(self, sub_weights_csr: sparse.csr_matrix, encode_idx: np.ndarray, decode_idx: np.ndarray,
                 config: SurrogateConfig, seed: int):
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

    def effective_weights(self) -> torch.Tensor:
        return self.sign * F.softplus(self.gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, dim) -> (batch,) predicted scalar f(x)."""
        cfg = self.config
        batch = x.shape[0]
        w = self.effective_weights()
        a = torch.sigmoid(self.leak_logit).unsqueeze(1)
        bias = self.bias.unsqueeze(1)
        h = torch.zeros(self.n, batch)
        inject = torch.zeros(self.n, batch)
        inject.index_add_(0, self.encode_idx, x.t() * cfg.input_scale)

        for _ in range(cfg.T):
            src = h.index_select(0, self.col)
            messages = src * w.unsqueeze(1)
            pre = torch.zeros(self.n, batch).index_add_(0, self.row, messages)
            pre = pre + inject
            h = (1 - a) * h + a * torch.tanh(pre + bias)

        decode_h = h.index_select(0, self.decode_idx)  # (n_readout, batch)
        return decode_h.t() @ self.readout_w + self.readout_b


def build_surrogate_training_set(dim: int, bounds: tuple[float, float], n_problems: int, n_points: int, seed: int):
    """Many randomly-offset Rastrigin instances (diversity across problem
    landscapes, same spirit as every other 2026-09-17 pilot), `n_points`
    random (x, f(x)) samples each -- straight regression targets, no
    direction/gradient needed."""
    from flyopt.benchmarks import rastrigin

    rng = np.random.default_rng(seed)
    lo, hi = bounds
    X, Y = [], []
    for _ in range(n_problems):
        offset = rng.uniform(lo * 0.4, hi * 0.4, size=dim)
        for _ in range(n_points):
            x = rng.uniform(lo, hi, size=dim)
            fx = rastrigin(x - offset)
            X.append(x)
            Y.append(fx)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def train_surrogate(brain: FlySurrogate, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float) -> list[float]:
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    X_t = torch.as_tensor(X)
    y_mean, y_std = float(Y.mean()), float(Y.std() + 1e-6)
    Y_t = torch.as_tensor((Y - y_mean) / y_std)  # normalized target: MSE comparable across problem instances
    curve = []
    for epoch in range(epochs):
        opt.zero_grad()
        pred = brain(X_t)
        loss = F.mse_loss(pred, Y_t)
        loss.backward()
        opt.step()
        curve.append(float(loss.item()))
    return curve
