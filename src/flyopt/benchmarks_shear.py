"""Beam transverse SHEAR design -- a SIXTH independent single-shot
physical/engineering benchmark (2026-09-24, user: "bir fiziksel problem
daha bul onu da test edelim, sineğin gücünü gösterelim"). Mechanically
distinct failure mode from all five prior tasks: transverse SHEAR
stress (average-shear approximation for a rectangular section,
tau = 1.5*V/(b*h)) -- a standard, separate structural-engineering check
from bending stress (beam_cost), and one with a DIFFERENT power-law
dependence on the cross-section (1/(b*h), linear in area, vs bending's
1/(b*h^2), buckling's 1/(b*h^3) via I, and deflection's also 1/(b*h^3)).

Deliberately mirrors beam_cost's exact structure (raw stress term + cost
term, SYMMETRIC (b,h) box) since that combination is the one proven to
need no rebalancing on the first try (beam's original delta=0.613 held
up without any correction, unlike column/vessel/deflection which all
needed a term-balance and/or scale fix -- see EXPERIMENTS.md 2026-09-24).
Verified via the now-standard pre-training diagnostic (invalid rate,
ratio-term objective share, finite-difference teacher-signal quality)
before any GPU time was spent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 2  # (b, h) -- rectangular cross-section width and depth, metres

LARGE_COST = 1e10
MAX_ASPECT = 6.0  # same practical slenderness guideline as beam_cost


@dataclass(frozen=True)
class ShearGeometry:
    shear_force: float   # N, design transverse shear force
    cost_weight: float   # lambda, material-cost coefficient

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([0.03, 0.03])
        hi = np.array([0.6, 0.6])
        return lo, hi


def random_geometry(rng: np.random.Generator) -> ShearGeometry:
    shear_force = float(rng.uniform(5_000.0, 60_000.0))
    cost_weight = float(rng.uniform(4e6, 3e7))
    return ShearGeometry(shear_force=shear_force, cost_weight=cost_weight)


def shear_cost(x: np.ndarray, geo: ShearGeometry) -> float:
    """x = (b, h). Objective to MINIMIZE: average transverse shear
    stress (1.5*V/(b*h), the standard rectangular-section approximation)
    plus a material-cost penalty (cost_weight * b * h) -- same
    "failure metric + cost" structure as beam_cost. Minimizing shear
    stress alone drives (b,h) to infinity; the cost term counters that."""
    b, h = float(x[0]), float(x[1])
    if b <= 1e-6 or h <= 1e-6:
        return LARGE_COST
    if max(h / b, b / h) > MAX_ASPECT:
        return LARGE_COST
    shear_stress = 1.5 * geo.shear_force / (b * h)
    material_cost = geo.cost_weight * b * h
    return shear_stress + material_cost


def sample_valid_point(geo: ShearGeometry, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    lo, hi = geo.param_bounds()
    x = rng.uniform(lo, hi)
    for _ in range(max_tries):
        if shear_cost(x, geo) < LARGE_COST:
            return x
        x = rng.uniform(lo, hi)
    print(f"[sample_valid_point] no valid (b,h) found in {max_tries} tries -- returning invalid point")
    return x
