"""Simply-supported beam DEFLECTION (serviceability) design -- a FIFTH
independent single-shot physical/engineering benchmark (2026-09-24, user:
"başka bir fiziksel problem bul"). Distinct DESIGN CRITERION from every
prior task, not just a distinct failure formula: this is a
serviceability/stiffness limit state (how much does it deflect under
load), not a strength/instability limit state (slope: limit-equilibrium
slip; beam: bending STRESS; vessel: hoop STRESS; column: elastic
buckling INSTABILITY). Standard structural-engineering distinction:
strength design keeps a member from breaking, serviceability design
keeps it from sagging/vibrating enough to be unusable, and the two
often govern in different regimes for the same member.

Classic textbook case: simply-supported rectangular beam, span L,
central point load P, max mid-span deflection delta = P*L^3/(48*E*I),
I = b*h^3/12. An allowable-deflection limit (L/250, a common code
value) gives a deflection RATIO (delta/allowable) that plays the same
"minimize this, it blows up as (b,h) shrink" role beam_cost's stress
term plays -- but with a DIFFERENT power-law sensitivity to h (cubic,
via I, vs beam_cost's quadratic h^2 in the stress denominator), so this
is not a re-skin of beam_cost's landscape. Deliberately reuses beam's
SYMMETRIC (b,h) box (same [0.03, 0.6] range for both dimensions) --
the vessel/column investigations found that asymmetric-scale parameter
pairs (like R,t) degrade the finite-difference teacher signal, so this
benchmark avoids that confound by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 2  # (b, h) -- rectangular cross-section width and depth, metres

LARGE_COST = 1e10

E_MODULUS = 200e9  # Pa, fixed material constant (steel), same convention as benchmarks_column.py
ALLOWABLE_RATIO = 250.0  # allowable deflection = span / 250 (a common serviceability code limit)


@dataclass(frozen=True)
class DeflectionGeometry:
    length: float       # m, simply-supported span
    load: float          # N, central point load
    cost_weight: float   # lambda, material-cost coefficient

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # 2026-09-24 recalibration (see random_geometry docstring, same issue and fix
        # as benchmarks_column.py's I~h^3 sensitivity): narrowed from [0.03,0.6] to a
        # tractable 27x range in h^3.
        lo = np.array([0.1, 0.1])
        hi = np.array([0.3, 0.3])
        return lo, hi


_REF_SIDE = 0.173  # geometric mean of the [0.1, 0.3] box, same calibration role as benchmarks_column.py


def random_geometry(rng: np.random.Generator) -> DeflectionGeometry:
    """2026-09-24 recalibration: deflection ~ 1/I ~ 1/h^3 has the exact
    same extreme sensitivity problem as benchmarks_column.py's Pcr (both
    involve I = b*h^3/12), and an initial fixed load range produced the
    identical degenerate symptom (material-cost term dominating >99.99%
    of the objective at typical sampled points, verified numerically --
    caught by comparing this task's diagnostic numbers to column's and
    finding them suspiciously near-identical, which traced back to both
    objectives collapsing to essentially the same trivial "minimize b*h"
    signal). Fixed the same way: narrowed box + load derived as a
    fraction of the allowable-deflection-implied load of a REFERENCE
    cross-section, rather than an independent fixed range."""
    length = float(rng.uniform(1.5, 6.0))
    i_ref = _REF_SIDE ** 4 / 12.0
    allowable = length / ALLOWABLE_RATIO
    # load fraction f such that deflection_ratio(ref) = f  <=>  load = f * allowable * 48 * E * i_ref / L^3
    load_ratio_target = float(rng.uniform(0.15, 0.85))
    load = load_ratio_target * allowable * 48.0 * E_MODULUS * i_ref / length ** 3
    cost_weight = float(rng.uniform(5.0, 40.0))
    return DeflectionGeometry(length=length, load=load, cost_weight=cost_weight)


def deflection_cost(x: np.ndarray, geo: DeflectionGeometry) -> float:
    """x = (b, h). Objective to MINIMIZE: deflection ratio (actual
    mid-span deflection / allowable deflection, dimensionless -- grows
    without bound as (b,h) shrink since I -> 0) plus a material-cost
    penalty (cost_weight * b * h), the same "failure metric + cost"
    structure as every other task in this family. Minimizing the
    deflection ratio alone drives (b,h) to infinity; the cost term
    counters that, giving a genuine interior optimum."""
    b, h = float(x[0]), float(x[1])
    if b <= 1e-6 or h <= 1e-6:
        return LARGE_COST
    i_section = b * h ** 3 / 12.0
    deflection = geo.load * geo.length ** 3 / (48.0 * E_MODULUS * i_section)
    allowable = geo.length / ALLOWABLE_RATIO
    deflection_ratio = deflection / allowable
    material_cost = geo.cost_weight * b * h
    return deflection_ratio + material_cost


def sample_valid_point(geo: DeflectionGeometry, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    lo, hi = geo.param_bounds()
    x = rng.uniform(lo, hi)
    for _ in range(max_tries):
        if deflection_cost(x, geo) < LARGE_COST:
            return x
        x = rng.uniform(lo, hi)
    print(f"[sample_valid_point] no valid (b,h) found in {max_tries} tries -- returning invalid point")
    return x
