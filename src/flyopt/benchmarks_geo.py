"""Slope stability critical slip surface search (2026-09-17, "gercek
uzamsal bir gorev dene" -- the user's own domain, geotechnical
engineering). Not one of PROTOCOL.md's pre-registered Faz 1 benchmarks
(sphere/rastrigin, benchmarks.py) -- a genuinely spatial global
optimization problem used for real in geotechnical literature (GA/PSO/SA
to find the critical slip surface), instead of another synthetic toy.

Ordinary Method of Slices (Fellenius), dry, single homogeneous soil
layer, circular trial slip surface -- the textbook-simplest limit
equilibrium method, chosen for a small, dependency-free implementation.
Minimize FS(xc, yc, R) over the trial-circle parameters; the *critical*
(most dangerous) slip surface is the one with the LOWEST factor of
safety, so this is a genuine minimization landscape, not a toy with a
known closed-form optimum -- there's no "offset trick"; local gradient
estimates (fly_proposer_scene.py's `_rays`/`_teacher_delta`) supply
supervision instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Trial-circle parameter bounds: (xc, yc, R). xc/yc in slope-height units,
# R in slope-height units -- see SlopeGeometry.bounds() for how these are
# scaled to a concrete slope.
DIM = 3
LARGE_FS = 50.0  # penalty for a trial circle that doesn't cut a valid slice mass


@dataclass(frozen=True)
class SlopeGeometry:
    """A simple two-break-point slope profile: flat ground at y=0 for
    x<=0, a planar slope of height H at angle beta (degrees) from (0,0)
    to (crest_x, H), flat ground at y=H for x>=crest_x."""

    height: float  # H (m)
    beta_deg: float  # slope angle from horizontal
    cohesion: float  # c (kPa)
    friction_deg: float  # phi (degrees)
    unit_weight: float  # gamma (kN/m^3)

    @property
    def crest_x(self) -> float:
        return self.height / np.tan(np.radians(self.beta_deg))

    def ground_y(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.where(x <= 0, 0.0, np.where(x >= self.crest_x, self.height, x * np.tan(np.radians(self.beta_deg))))
        return y

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) for (xc, yc, R): center swept over/around the slope
        face, radius from a shallow toe-circle to a deep one -- wide
        enough that both trivial (too-shallow, invalid) and unreasonably
        deep circles are explorable, forcing the optimizer to find the
        genuine critical one instead of the search space's edge."""
        lo = np.array([-0.5 * self.height, self.height * 0.3, 0.3 * self.height])
        hi = np.array([self.crest_x + 1.5 * self.height, self.height * 2.5, 3.0 * self.height])
        return lo, hi


def factor_of_safety(params: np.ndarray, geometry: SlopeGeometry, n_slices: int = 24) -> float:
    """Fellenius (Ordinary Method of Slices), dry, single layer.

    FS = sum(c*dl + W*cos(a)*tan(phi)) / sum(W*sin(a))

    Returns LARGE_FS (a bad, unphysical value -- never a spurious minimum
    since it's a fixed ceiling) if the trial circle does not produce a
    valid slice mass: doesn't intersect the ground surface in two points,
    the arc lies above ground, or the resulting FS would be non-finite.
    """
    xc, yc, R = params
    if R <= 0:
        return LARGE_FS

    xs = np.linspace(geometry.param_bounds()[0][0] - geometry.height, geometry.param_bounds()[1][0] + geometry.height, 2000)
    ground = geometry.ground_y(xs)
    under_circle = (xs - xc) ** 2 <= R * R
    if not under_circle.any():
        return LARGE_FS
    circle_y = np.full_like(xs, np.nan)
    circle_y[under_circle] = yc - np.sqrt(np.maximum(R * R - (xs[under_circle] - xc) ** 2, 0.0))
    below = under_circle & (circle_y < ground - 1e-6)
    if not below.any():
        return LARGE_FS
    idx = np.flatnonzero(below)
    x_left, x_right = xs[idx[0]], xs[idx[-1]]
    if x_right - x_left < 0.1 * geometry.height:
        return LARGE_FS

    edges = np.linspace(x_left, x_right, n_slices + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dx = np.diff(edges)
    y_top = geometry.ground_y(mids)
    under = (mids - xc) ** 2 <= R * R
    if not under.all():
        return LARGE_FS
    y_bot = yc - np.sqrt(R * R - (mids - xc) ** 2)
    h = y_top - y_bot
    if np.any(h <= 0):
        return LARGE_FS

    sin_a = (mids - xc) / R
    sin_a = np.clip(sin_a, -0.999, 0.999)
    alpha = np.arcsin(sin_a)
    cos_a = np.cos(alpha)
    dl = dx / np.maximum(cos_a, 1e-3)

    W = geometry.unit_weight * h * dx
    phi = np.radians(geometry.friction_deg)
    resisting = geometry.cohesion * dl + W * cos_a * np.tan(phi)
    driving = W * sin_a

    denom = float(np.sum(driving))
    if denom <= 1e-6:
        return LARGE_FS
    fs = float(np.sum(resisting) / denom)
    if not np.isfinite(fs) or fs <= 0:
        return LARGE_FS
    return min(fs, LARGE_FS)


def random_geometry(rng: np.random.Generator) -> SlopeGeometry:
    """Realistic-ish ranges for a routine (non-extreme) cut/fill slope,
    wide enough to give genuinely different landscapes per instance."""
    return SlopeGeometry(
        height=rng.uniform(8.0, 25.0),
        beta_deg=rng.uniform(25.0, 45.0),
        cohesion=rng.uniform(5.0, 40.0),
        friction_deg=rng.uniform(15.0, 35.0),
        unit_weight=rng.uniform(17.0, 21.0),
    )


def sample_valid_point(geometry: SlopeGeometry, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    """Rejection-sample a trial circle with a real (non-penalty) factor of
    safety. Fixes the 2026-09-18 bug (EXPERIMENTS.md): a plain
    `rng.uniform(lo, hi)` over `param_bounds()` lands on an invalid
    circle (LARGE_FS) 43.6% of the time (measured, n=500) -- and since
    LARGE_FS is a flat constant, a search that starts there sees a
    zero local gradient in every direction and can never escape. This
    does not fully remove the risk of wandering into an invalid region
    mid-search (a circle near the valid/invalid boundary still has a
    real gradient pointing back, so that case is far less severe), but
    it removes the dominant failure mode: starting there outright.
    Falls back to the last sampled point (with a print) if it never
    finds a valid one in `max_tries` -- should not happen in practice
    at the measured ~56% valid rate."""
    lo, hi = geometry.param_bounds()
    x = rng.uniform(lo, hi)
    for _ in range(max_tries):
        if factor_of_safety(x, geometry) < LARGE_FS:
            return x
        x = rng.uniform(lo, hi)
    print(f"[sample_valid_point] no valid circle found in {max_tries} tries for this geometry -- returning invalid point")
    return x
