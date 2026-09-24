"""Random-MLP control substrate (PROTOCOL.md 2.2, 3): a fixed, untrained,
randomly-initialized feedforward network with a parameter count matched to
a reference graph's edge count. This is the control that isolates "any
fixed random recurrent-ish substrate helps" from "this connectome's
specific topology helps" — the Fly-Controller variant lives or dies on
whether it beats this.

Not a Substrate for LIF spiking dynamics; it operates directly on
continuous stimulus/activity vectors. Recurrence is the minimal form
(previous output fed back as part of the input) so it belongs to the same
broad class of "stateful fixed dynamical system" as the graph substrates,
without importing any connectome structure.
"""
from __future__ import annotations

import numpy as np


class RandomMLPState:
    __slots__ = ("prev_output",)

    def __init__(self, prev_output: np.ndarray):
        self.prev_output = prev_output


class RandomMLPSubstrate:
    def __init__(self, n_units: int, n_params_target: int, seed_init: int = 0):
        # hidden size chosen so that total weight count (input->hidden,
        # hidden->hidden recurrent, hidden->output) is close to n_params_target
        rng = np.random.default_rng(seed_init)
        hidden = max(1, int(round(n_params_target / max(1, 3 * n_units))))
        self._n_units = n_units
        self._hidden = hidden
        scale = 1.0 / np.sqrt(max(1, n_units))
        self.W_in = rng.normal(0, scale, size=(hidden, n_units))
        self.W_rec = rng.normal(0, scale, size=(hidden, n_units))
        self.W_out = rng.normal(0, scale, size=(n_units, hidden))

    @property
    def n_units(self) -> int:
        return self._n_units

    @property
    def n_params(self) -> int:
        return self.W_in.size + self.W_rec.size + self.W_out.size

    def reset(self, seed: int) -> RandomMLPState:
        rng = np.random.default_rng(seed)
        return RandomMLPState(prev_output=rng.normal(0, 0.01, size=self._n_units))

    def step(
        self,
        stimulus: np.ndarray,
        state: RandomMLPState,
        reward: float | None = None,
    ) -> tuple[np.ndarray, RandomMLPState]:
        h = np.tanh(self.W_in @ stimulus + self.W_rec @ state.prev_output)
        out = np.tanh(self.W_out @ h)
        return out, RandomMLPState(prev_output=out)
