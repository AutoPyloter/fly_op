"""Core Substrate/Proposer interfaces (PROTOCOL.md section 3).

Every variant (Fly-Proposer, Fly-Controller, ...) is a wrapper around a
Substrate. Do not add variant-specific methods here — if a variant needs
something this interface can't express, the interface is wrong, not the
variant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

State = Any  # opaque, substrate-defined


@runtime_checkable
class Substrate(Protocol):
    def reset(self, seed: int) -> State:
        """Initialize/reinitialize internal dynamics state for a fresh run."""
        ...

    def step(
        self,
        stimulus: np.ndarray,
        state: State,
        reward: Optional[float] = None,
    ) -> tuple[np.ndarray, State]:
        """Advance dynamics by one step given a stimulus, return (activity, new_state).

        `reward` is None for substrates without plasticity (the default for
        every variant except Fly-RL, PROTOCOL.md 2.5). Substrates that ignore
        plasticity must accept and ignore it rather than erroring.
        """
        ...

    @property
    def n_units(self) -> int:
        """Number of dynamical units (neurons) in the substrate."""
        ...


@dataclass
class SearchContext:
    """Read-only info a Proposer may use to decide what to propose next.

    Kept minimal on purpose — extend only when a variant demonstrably needs
    a field, not speculatively.
    """

    iteration: int
    budget_total: int
    budget_used: int
    best_x: np.ndarray
    best_fx: float
    history_fx: list[float] = field(default_factory=list)


@runtime_checkable
class Proposer(Protocol):
    def propose(self, x: np.ndarray, ctx: SearchContext) -> np.ndarray:
        """Produce a new candidate from the current point x."""
        ...

    def tell(self, x: np.ndarray, fx: float, improved: bool) -> None:
        """Feedback after evaluation. No-op for proposers without adaptation."""
        ...
