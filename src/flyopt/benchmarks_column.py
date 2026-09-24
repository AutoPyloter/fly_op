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
        lo = np.array([0.03, 0.03])
        hi = np.array([0.6, 0.6])
        return lo, hi


def random_geometry(rng: np.random.Generator) -> ColumnGeometry:
    length = float(rng.uniform(1.5, 6.0))
    load = float(rng.uniform(5_000.0, 200_000.0))
    cost_weight = float(rng.uniform(4e6, 3e7))
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
