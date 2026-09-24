"""Thin-walled pressure vessel sizing -- a THIRD independent physical/
engineering optimization benchmark (2026-09-24), added to test whether
the "narrow task class" finding (single-shot spatial regression on
physical/engineering objectives, EXPERIMENTS.md 2026-09-17..24) rests on
genuinely more than the two examples tested so far (slope stability:
geotechnical limit-equilibrium; beam design: bending stress). Same DIM=2,
param_bounds/random_geometry/objective(x, geo) convention as
benchmarks_geo.py and benchmarks_structural.py, directly reusable with
fly_proposer_scene.py's _rays / _teacher_delta -- but a mechanically
distinct failure mode (thin-wall hoop stress under internal pressure,
not bending or slip-surface stability).

Classic textbook problem: a thin-walled cylindrical pressure vessel,
internal pressure P, radius R, wall thickness t. Hoop (circumferential)
stress sigma = P*R/t (thin-wall approximation, valid for R/t large)
alone drives t to infinity (zero stress at infinite thickness), so a
material-cost term (proportional to wall cross-sectional area, ~R*t) is
added, creating a genuine smooth interior minimum -- the same role
beam_cost's cost_weight*b*h term plays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 2  # (R, t) -- vessel radius and wall thickness, metres

LARGE_COST = 1e10  # invalid/degenerate (b,h)-style penalty, matches beam_cost's convention

MIN_THINWALL_RATIO = 8.0  # R/t must exceed this for the thin-wall hoop-stress approximation to be
                            # valid and to stay out of the thick-wall/buckling regime -- standard
                            # pressure-vessel-design guideline (ASME-style rule of thumb is R/t > 10;
                            # 8 used here to keep the box-constrained search space non-trivially large)


@dataclass(frozen=True)
class VesselGeometry:
    pressure: float      # Pa, internal gauge pressure
    cost_weight: float   # lambda, material-cost coefficient

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([0.05, 0.001])
        hi = np.array([1.2, 0.08])
        return lo, hi


def random_geometry(rng: np.random.Generator) -> VesselGeometry:
    """2026-09-24 recalibration (applying the same lesson learned fixing
    benchmarks_column.py's term-balance bug -- see EXPERIMENTS.md
    2026-09-24 -- to this task's original, unexplained FAIL): the
    original cost_weight range [5e6, 8e7] left the hoop-STRESS term
    dominating >97% of the objective at typical sampled points (opposite
    imbalance from column's original bug, same underlying flaw: one term
    trivially dominates, making the task "just minimize/maximize one
    variable" regardless of connectome structure). Verified numerically:
    cost_weight=[5e8, 4e9] brings the stress-term share down to a median
    ~40% (real tension between the two terms). The original vessel_cost
    delta=-0.031 FAIL result is RETRACTED as invalid/degenerate pending
    a rerun with this calibration -- not treated as a genuine negative
    result."""
    pressure = float(rng.uniform(2e5, 3e6))       # 0.2-3.0 MPa gauge
    cost_weight = float(rng.uniform(5e8, 4e9))
    return VesselGeometry(pressure=pressure, cost_weight=cost_weight)


def vessel_cost(x: np.ndarray, geo: VesselGeometry) -> float:
    """x = (R, t). Objective to MINIMIZE: hoop stress (P*R/t, thin-wall
    approximation) plus a material-cost penalty (cost_weight * R * t,
    proportional to wall cross-sectional material per unit length).
    Minimizing stress alone drives t to infinity; the cost term counters
    that. The R/t thin-wall-validity floor plays the same role
    beam_cost's MAX_ASPECT cap does -- it bounds the search away from a
    regime the simple formula doesn't model, giving a genuine interior
    optimum. Invalid points return a large finite penalty (never
    inf/nan)."""
    R, t = float(x[0]), float(x[1])
    if R <= 1e-6 or t <= 1e-6:
        return LARGE_COST
    if R / t < MIN_THINWALL_RATIO:
        return LARGE_COST
    hoop_stress = geo.pressure * R / t
    material_cost = geo.cost_weight * R * t
    return hoop_stress + material_cost


def sample_valid_point(geo: VesselGeometry, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    """Rejection-sample a trial (R, t) with a real (non-penalty) cost --
    same guard pattern as benchmarks_geo.py / benchmarks_structural.py."""
    lo, hi = geo.param_bounds()
    x = rng.uniform(lo, hi)
    for _ in range(max_tries):
        if vessel_cost(x, geo) < LARGE_COST:
            return x
        x = rng.uniform(lo, hi)
    print(f"[sample_valid_point] no valid (R,t) found in {max_tries} tries -- returning invalid point")
    return x
