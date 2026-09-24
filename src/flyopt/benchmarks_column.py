"""Euler column buckling -- a FOURTH independent single-shot physical/
engineering benchmark (2026-09-24, user: "bir fiziksel problem ara onu da
test edelim"), added to build statistical power on the "how many physical
tasks show the advantage" question (currently 2 PASS -- slope stability,
beam design -- out of 3 tested, vessel design an unexplained FAIL, see
EXPERIMENTS.md).

Mechanically distinct failure mode from all three prior tasks: elastic
BUCKLING INSTABILITY under axial compression (a stability/eigenvalue
phenomenon), not bending stress (beam), hoop stress (vessel), or
limit-equilibrium slip (slope). Deliberately mirrors beam_cost's (b, h)
rectangular-section parametrisation and bound RANGE (same [0.03, 0.6]
box for both dimensions) rather than vessel's asymmetric (R, t) pair --
the vessel investigation (EXPERIMENTS.md 2026-09-24) found a parameter-
scale mismatch degraded the finite-difference teacher signal there, so
this benchmark avoids that specific confound by construction: both
design variables share the same scale/box, exactly like the already-
validated beam task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 2  # (b, h) -- rectangular column cross-section width and depth, metres

LARGE_COST = 1e10

E_MODULUS = 200e9  # Pa, fixed material constant (steel), analogous to beam_cost's implicit material assumption


@dataclass(frozen=True)
class ColumnGeometry:
    length: float       # m, pinned-pinned effective column length
    load: float          # N, axial compressive load
    cost_weight: float   # lambda, material-cost coefficient

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # 2026-09-24 recalibration (see random_geometry docstring): narrowed from the
        # original [0.03, 0.6] (an 8000x range in h^3, the buckling term's sensitivity)
        # to a 27x range in h^3, where a balanced load/cost_weight choice actually exists.
        lo = np.array([0.1, 0.1])
        hi = np.array([0.3, 0.3])
        return lo, hi


_REF_SIDE = 0.173  # geometric mean of the [0.1, 0.3] box -- a "typical" cross-section, used only to
                    # calibrate the load range below, not a hidden default answer


def random_geometry(rng: np.random.Generator) -> ColumnGeometry:
    """2026-09-24 recalibration: Euler critical load Pcr is extremely
    sensitive to (b,h,L) (varies by ~5 orders of magnitude across the
    ORIGINAL [0.03,0.6] box), so a fixed load range made the load
    negligible next to Pcr for most sampled cross-sections -- the
    material-cost term then dominated the objective almost completely
    (ratio-term share <0.01% at typical points, verified numerically),
    making the task trivial (just "minimize b*h", a linear/bilinear term
    with an exact finite-difference gradient regardless of step size)
    independent of connectome structure. This produced the FAIL initially
    reported for this task -- retracted as an invalid/degenerate test,
    not a genuine negative result (see EXPERIMENTS.md 2026-09-24).

    Fixed in two parts: (1) the (b,h) box was narrowed (see param_bounds)
    to keep h^3's range tractable, and (2) load is now derived as a
    FRACTION of the critical load of a REFERENCE cross-section (_REF_SIDE,
    the box's geometric mean) at the sampled length, rather than an
    independent fixed range -- ties load magnitude to what's structurally
    meaningful for that span. Verified numerically post-fix: median
    ratio-term share of the objective ~39% (was <0.01%), i.e. both terms
    now genuinely compete across most of the training distribution."""
    length = float(rng.uniform(1.5, 6.0))
    i_ref = _REF_SIDE ** 4 / 12.0
    p_ref = float(np.pi ** 2 * E_MODULUS * i_ref / length ** 2)
    load = float(rng.uniform(0.15, 0.85)) * p_ref
    cost_weight = float(rng.uniform(5.0, 40.0))
    return ColumnGeometry(length=length, load=load, cost_weight=cost_weight)


def column_cost(x: np.ndarray, geo: ColumnGeometry) -> float:
    """x = (b, h). Objective to MINIMIZE: applied-load / Euler-critical-
    load ratio (dimensionless, an inverse buckling factor of safety --
    grows without bound as the section approaches its critical load)
    plus a material-cost penalty (cost_weight * cross-sectional area),
    the same "failure metric + cost" structure as beam_cost and
    vessel_cost. Buckles about the weak axis: I_min = min(b*h^3, h*b^3)/12.
    Minimizing the load ratio alone drives (b,h) to infinity; the cost
    term counters that, giving a genuine interior optimum."""
    b, h = float(x[0]), float(x[1])
    if b <= 1e-6 or h <= 1e-6:
        return LARGE_COST
    i_min = min(b * h ** 3, h * b ** 3) / 12.0
    p_crit = (np.pi ** 2) * E_MODULUS * i_min / (geo.length ** 2)
    if p_crit <= 1e-6:
        return LARGE_COST
    load_ratio = geo.load / p_crit
    material_cost = geo.cost_weight * b * h
    return load_ratio + material_cost


def sample_valid_point(geo: ColumnGeometry, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    lo, hi = geo.param_bounds()
    x = rng.uniform(lo, hi)
    for _ in range(max_tries):
        if column_cost(x, geo) < LARGE_COST:
            return x
        x = rng.uniform(lo, hi)
    print(f"[sample_valid_point] no valid (b,h) found in {max_tries} tries -- returning invalid point")
    return x
