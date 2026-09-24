"""The one Substrate implementation every graph-based variant shares.

FlyWireSubstrate, ERNullSubstrate, DegreePreservingNullSubstrate,
WeightShuffleNullSubstrate and SignFlipNullSubstrate are all this same
class, constructed with a different adjacency (see factory.py). This is the
concrete embodiment of PROTOCOL.md section 3's "tek arayüz" rule — do not
give any of them their own step()/reset() logic.
"""
from __future__ import annotations

import numpy as np

from flyopt.sim.lif_vectorized import LIFParams, LIFState, SparseLIFEngine


class SparseRecurrentSubstrate:
    def __init__(self, weights_csr, params: LIFParams = LIFParams()):
        self._engine = SparseLIFEngine(weights_csr, params)

    def reset(self, seed: int) -> LIFState:
        return self._engine.reset(seed)

    def step(
        self, stimulus: np.ndarray, state: LIFState, reward: float | None = None
    ) -> tuple[np.ndarray, LIFState]:
        # No plasticity in this substrate; reward is accepted and ignored,
        # per the Substrate protocol contract.
        return self._engine.step(stimulus, state)

    @property
    def n_units(self) -> int:
        return self._engine.n_units
