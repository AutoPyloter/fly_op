"""(1+1) elitist hill-climber — the one search loop shared by every
substrate (PROTOCOL.md section 3). The substrate wrapped inside `proposer`
is the only thing that may differ between a real-connectome run and a
null-model run; this function never sees which one it got.
"""
from __future__ import annotations

import numpy as np

from flyopt.substrate import Proposer, SearchContext


def run_hillclimb(
    proposer: Proposer,
    objective,
    dim: int,
    bounds: tuple[float, float],
    budget: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    x = rng.uniform(lo, hi, size=dim)
    fx = objective(x)
    best_fx = fx

    for it in range(budget):
        ctx = SearchContext(iteration=it, budget_total=budget, budget_used=it, best_x=x, best_fx=best_fx)
        candidate = np.clip(proposer.propose(x, ctx), lo, hi)
        f_cand = objective(candidate)
        improved = f_cand < fx
        proposer.tell(candidate, f_cand, improved)
        if improved:
            x, fx = candidate, f_cand
        if fx < best_fx:
            best_fx = fx

    return best_fx
