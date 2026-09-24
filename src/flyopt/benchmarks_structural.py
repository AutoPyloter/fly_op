"""Cantilever beam cross-section design -- a second, independent
physical/engineering optimization benchmark (2026-09-21, user:
"benchmarklari denedin mi? onlari yapabilirsin" -- Gemini's suggested
FEA/structural surrogate idea, gemini_arastirma_tavsiyeleri.md Bolum
4.2). Analogous in spirit to benchmarks_geo.py's slope-stability task
(same DIM=2, same param_bounds/random_geometry/objective(x, geo)
convention, directly reusable with fly_proposer_scene.py's _rays /
_teacher_delta) but a genuinely different physical domain -- structural
mechanics, not geotechnical -- to test whether the connectome's measured
advantage (EXPERIMENTS.md 2026-09-17..21) generalizes across physical
TASK types, not just within slope stability specifically.

Classic textbook structural-optimization problem: a cantilever beam,
length L, tip point load P, rectangular cross-section (width b, height
h). Bending stress sigma = 6*P*L / (b*h^2) alone drives b,h to infinity
(zero stress at infinite cross-section), so a material-cost term
(proportional to cross-sectional area b*h) is added, creating a genuine
smooth interior minimum -- the same "well-posed 2D landscape" role
benchmarks_geo.py's factor-of-safety plays for slope stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 2  # (b, h) -- cross-section width and height, metres

LARGE_COST = 1e10  # returned for invalid (non-positive/over-slender) cross-sections -- must exceed any realistic valid cost (up to ~1.5e7 at this scale)


MAX_ASPECT = 6.0  # h/b or b/h beyond this is impractical (buckling/serviceability) -- standard rectangular-beam guideline


@dataclass(frozen=True)
class BeamGeometry:
    length: float       # m, cantilever span
    load: float          # N, point load at the free end
    cost_weight: float   # lambda, material-cost coefficient (N/m^3-ish)

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([0.03, 0.03])
        hi = np.array([0.6, 0.6])
        return lo, hi


def random_geometry(rng: np.random.Generator) -> BeamGeometry:
    length = float(rng.uniform(1.5, 6.0))
    load = float(rng.uniform(2_000.0, 25_000.0))
    cost_weight = float(rng.uniform(4e6, 3e7))
    return BeamGeometry(length=length, load=load, cost_weight=cost_weight)


def beam_cost(x: np.ndarray, geo: BeamGeometry) -> float:
    """x = (b, h). Objective to MINIMIZE: max bending stress (6*P*L /
    (b*h^2), standard rectangular-section cantilever formula) plus a
    material-cost penalty (cost_weight * cross-sectional area).

    Minimizing stress alone drives (b,h) to infinity (zero stress at
    infinite section) -- the material-cost term counters that, but
    because stress falls off much faster in h than in b (h^2 vs b^1),
    an UNCONSTRAINED version still pushes b toward the domain's lower
    edge (real engineering fact: deep, thin sections are bending-
    efficient). A slenderness/aspect-ratio cap (MAX_ASPECT, a standard
    practical guideline against lateral-torsional buckling) bounds this,
    giving b a genuine interior optimum for typical (load, length,
    cost_weight) draws while h may still sit near its own box bound --
    both are smooth, gradient-bearing outcomes (not the flat/degenerate
    zero-gradient trap found and fixed in benchmarks_geo.py's slope
    task). Invalid or over-slender cross-sections return a large finite
    penalty (never inf/nan), matching that same LARGE_FS convention.
    """
    b, h = float(x[0]), float(x[1])
    if b <= 1e-6 or h <= 1e-6:
        return LARGE_COST
    if max(h / b, b / h) > MAX_ASPECT:
        return LARGE_COST
    stress = 6.0 * geo.load * geo.length / (b * h ** 2)
    material_cost = geo.cost_weight * b * h
    return stress + material_cost


def sample_valid_point(geo: BeamGeometry, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    """Rejection-sample a trial (b, h) with a real (non-penalty) cost --
    same fix pattern as benchmarks_geo.py's sample_valid_point, applied
    pre-emptively here (measured ~10.9% invalid rate at random points,
    well under the 43.6% that caused that earlier bug, but the failure
    mode -- starting a search at a flat LARGE_COST point with zero local
    gradient -- is identical, so the same rejection-sampling guard is
    used from the start rather than discovered after the fact)."""
    lo, hi = geo.param_bounds()
    x = rng.uniform(lo, hi)
    for _ in range(max_tries):
        if beam_cost(x, geo) < LARGE_COST:
            return x
        x = rng.uniform(lo, hi)
    print(f"[sample_valid_point] no valid cross-section found in {max_tries} tries -- returning invalid point")
    return x
