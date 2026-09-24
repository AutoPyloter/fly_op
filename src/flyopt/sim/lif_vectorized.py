"""Vectorized sparse LIF dynamics (torch), used inside every graph-based
Substrate (PROTOCOL.md section 5, Faz 0: "PyTorch/JAX ile vektörize sparse
matmul versiyonu").

This module is deliberately dumb: it knows nothing about connectomes, null
models, or optimization. It takes a fixed weighted adjacency and simulates
leaky integrate-and-fire dynamics for a fixed number of steps given an input
current at each step. `lif_brian2.py` re-implements the same equations with
Brian2 as an independent reference for the equivalence test in
tests/test_lif_equivalence.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class LIFParams:
    tau_m_ms: float = 20.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    refractory_ms: float = 2.0
    dt_ms: float = 1.0


@dataclass
class LIFState:
    v: torch.Tensor  # membrane potential, shape (n_units,)
    refractory_remaining: torch.Tensor  # shape (n_units,)
    last_spikes: torch.Tensor  # spikes emitted at the previous step (1-step synaptic delay)


class SparseLIFEngine:
    """LIF network driven by a fixed weighted sparse adjacency.

    weights[i, j] = synaptic weight from unit j onto unit i (i.e. the
    adjacency is stored pre-transposed for `W @ spikes`), already signed
    (positive = excitatory, negative = inhibitory).
    """

    def __init__(self, weights_csr, params: LIFParams = LIFParams()):
        # CSR, not COO: measured ~40x faster for torch.sparse.mm at FlyWire
        # scale (0.32s -> 0.008s per step, 139k units / 15M edges, CPU) —
        # COO sparse.mm was the actual Faz 0 cost bottleneck, not the
        # connectome's size per se. See EXPERIMENTS.md.
        self.n_units = weights_csr.shape[0]
        coo_scipy = weights_csr.tocoo()
        indices = torch.tensor(np.vstack([coo_scipy.row, coo_scipy.col]), dtype=torch.long)
        values = torch.tensor(coo_scipy.data, dtype=torch.float32)
        coo = torch.sparse_coo_tensor(
            indices, values, size=(self.n_units, self.n_units)
        ).coalesce()
        self.weights = coo.to_sparse_csr()
        self.params = params

    def reset(self, seed: int) -> LIFState:
        gen = torch.Generator().manual_seed(seed)
        v0 = torch.rand(self.n_units, generator=gen) * self.params.v_threshold
        return LIFState(
            v=v0,
            refractory_remaining=torch.zeros(self.n_units),
            last_spikes=torch.zeros(self.n_units),
        )

    def step(
        self, stimulus: np.ndarray, state: LIFState
    ) -> tuple[np.ndarray, LIFState]:
        p = self.params
        input_current = torch.as_tensor(stimulus, dtype=torch.float32)
        active = state.refractory_remaining <= 0

        recurrent_input = torch.sparse.mm(
            self.weights, state.last_spikes.unsqueeze(1)
        ).squeeze(1)

        dv = (p.dt_ms / p.tau_m_ms) * (
            -(state.v - p.v_rest) + input_current + recurrent_input
        )
        v_candidate = state.v + dv
        v_new = torch.where(active, v_candidate, state.v)

        spikes = (v_new >= p.v_threshold).float()
        v_new = torch.where(spikes.bool(), torch.full_like(v_new, p.v_reset), v_new)

        refractory_remaining = torch.where(
            spikes.bool(),
            torch.full_like(state.refractory_remaining, p.refractory_ms),
            (state.refractory_remaining - p.dt_ms).clamp(min=0.0),
        )

        new_state = LIFState(
            v=v_new, refractory_remaining=refractory_remaining, last_spikes=spikes
        )
        return spikes.numpy(), new_state
