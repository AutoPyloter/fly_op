"""Shallow-foundation bearing-capacity design -- a SEVENTH independent
single-shot physical/engineering benchmark (2026-09-24, user: "yeni bir
fiziksel deneyi olabilir ... neyi araştırmak istiyorsun yaz"). Mechanically
distinct failure mode from all six prior tasks: general shear bearing
failure under a strip footing (Terzaghi bearing-capacity equation),
the classic geotechnical companion to benchmarks_geo.py's slope-stability
task -- same domain (geotechnical engineering, the user's own field) but
a completely different limit state (punching/general-shear bearing
capacity vs. rotational slip-surface stability).

x = (B, Df): footing width and embedment depth, metres. Objective =
utilization ratio (applied pressure / ultimate bearing capacity) + a
material/excavation cost term (cost_weight * B * Df) -- the same proven
"failure-utilization-ratio + cost" structure as column_cost/shear_cost,
the two tasks in this family that passed calibration on the first try.

Calibrated BEFORE any GPU time was spent, per the project's mandatory
pre-training diagnostic (see README.md Sec. 11): invalid-point rate 0.0%,
utilization-term objective share median ~36% (p25=18%, p75=61%) across
4000 sampled points at cost_weight in [0.05, 0.5], and finite-difference
teacher-signal cosine similarity vs. a near-exact local gradient (radius
1e-6) of 1.0000 +/- 0.0000 at RAY_RADIUS=0.01 across 800 sampled points.
Load is derived from each problem's own soil strength (evaluated at the
box's reference geometry), the same anti-term-imbalance technique used by
benchmarks_column.py, applied proactively this time instead of after a
failed first attempt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 2  # (B, Df) -- footing width and embedment depth, metres

LARGE_COST = 1e10
MAX_ASPECT = 8.0  # B/Df or Df/B beyond this is an impractical footing proportion

_B_LO, _B_HI = 0.5, 3.0
_DF_LO, _DF_HI = 0.3, 2.5
_REF_B = float(np.sqrt(_B_LO * _B_HI))
_REF_DF = float(np.sqrt(_DF_LO * _DF_HI))


def _bearing_factors(friction_deg: float) -> tuple[float, float, float]:
    """Terzaghi/Vesic general bearing-capacity factors Nc, Nq, Ngamma."""
    phi = np.radians(friction_deg)
    Nq = float(np.exp(np.pi * np.tan(phi)) * (np.tan(np.radians(45.0) + phi / 2.0)) ** 2)
    Nc = (Nq - 1.0) / np.tan(phi)
    Ngamma = 2.0 * (Nq + 1.0) * np.tan(phi)
    return Nc, Nq, Ngamma


def _q_ult(B: float, Df: float, cohesion: float, friction_deg: float, unit_weight: float) -> float:
    Nc, Nq, Ng = _bearing_factors(friction_deg)
    return cohesion * Nc + unit_weight * Df * Nq + 0.5 * unit_weight * B * Ng


@dataclass(frozen=True)
class BearingGeometry:
    cohesion: float      # c, kPa
    friction_deg: float  # phi, degrees
    unit_weight: float   # gamma, kN/m^3
    load: float           # P, kN/m (line load on the strip footing)
    cost_weight: float    # lambda, material/excavation-cost coefficient

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([_B_LO, _DF_LO])
        hi = np.array([_B_HI, _DF_HI])
        return lo, hi


def random_geometry(rng: np.random.Generator) -> BearingGeometry:
    cohesion = float(rng.uniform(5.0, 40.0))
    friction_deg = float(rng.uniform(20.0, 35.0))
    unit_weight = float(rng.uniform(16.0, 20.0))
    q_ref = _q_ult(_REF_B, _REF_DF, cohesion, friction_deg, unit_weight)
    util_target = float(rng.uniform(0.15, 0.85))
    load = util_target * q_ref * _REF_B
    cost_weight = float(rng.uniform(0.05, 0.5))
    return BearingGeometry(cohesion=cohesion, friction_deg=friction_deg, unit_weight=unit_weight,
                            load=load, cost_weight=cost_weight)


def bearing_cost(x: np.ndarray, geo: BearingGeometry) -> float:
    """x = (B, Df). Objective to MINIMIZE: bearing-capacity utilization
    ratio (applied pressure / ultimate capacity) plus a material/excavation
    cost penalty (cost_weight * B * Df). Minimizing utilization alone
    drives B to infinity; the cost term counters that."""
    B, Df = float(x[0]), float(x[1])
    if B <= 1e-6 or Df <= 1e-6:
        return LARGE_COST
    if max(B / Df, Df / B) > MAX_ASPECT:
        return LARGE_COST
    q_ult = _q_ult(B, Df, geo.cohesion, geo.friction_deg, geo.unit_weight)
    if q_ult <= 1e-9:
        return LARGE_COST
    q_applied = geo.load / B
    utilization = q_applied / q_ult
    material_cost = geo.cost_weight * B * Df
    return utilization + material_cost


def sample_valid_point(geo: BearingGeometry, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    lo, hi = geo.param_bounds()
    x = rng.uniform(lo, hi)
    for _ in range(max_tries):
        if bearing_cost(x, geo) < LARGE_COST:
            return x
        x = rng.uniform(lo, hi)
    print(f"[sample_valid_point] no valid (B,Df) found in {max_tries} tries -- returning invalid point")
    return x
