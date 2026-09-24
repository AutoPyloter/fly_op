"""Alternative swarm organizations to classic global-best PSO (2026-09-18,
"alternatif olarak suru davranisi nasil organize ederiz"). Three
biologically-motivated variants, no central "gbest" bookkeeping unless
noted -- each fly only senses what a real fly could plausibly sense
(nearby others, a scent-like signal), not an omniscient shared scalar.

  chemotaxis   : each fly senses OTHER flies as a "scent field" -- angle,
                 distance, and fitness-as-strength (lower fitness =
                 stronger "smell", like fermenting fruit) -- exactly the
                 same panoramic construction as fly_swarm_social.py's
                 `_social_rays`, reused here. No gbest at all: pull comes
                 purely from the locally-sensed scent gradient.
  lek          : a few TOP-K performing flies act as "attractive mates";
                 every other fly is pulled toward its NEAREST attractive
                 one (not necessarily the single global best) -- a
                 multi-modal, decentralized analogue of PSO's single-
                 attractor pull, motivated by real lekking/mate-choice
                 behaviour (multiple attractive individuals, not one).
  ring_lbest   : standard PSO literature variant (Kennedy & Eberhart's
                 lbest topology): each particle only knows the best
                 position found by its two ring-neighbours, not the
                 whole swarm -- slower convergence, less premature
                 collapse, included as a literature-standard control.
"""
from __future__ import annotations

import numpy as np


def chemotaxis_step(x: np.ndarray, fx: np.ndarray, i: int) -> np.ndarray:
    """Pull for particle i towards the strongest-smelling nearby particle
    (lower fx = stronger smell), weighted by 1/(1+distance) -- no global
    best used anywhere."""
    others = np.array([j for j in range(len(x)) if j != i])
    rel = x[others] - x[i]
    dist = np.linalg.norm(rel, axis=1) + 1e-6
    quality = fx[i] - fx[others]  # positive if other is better (lower fx)
    quality = np.clip(quality, 0, None)
    strength = quality / (1.0 + dist)
    if strength.sum() < 1e-9:
        return np.zeros_like(x[i])
    weights = strength / strength.sum()
    target = (weights[:, None] * x[others]).sum(axis=0)
    return target - x[i]


def lek_pull(x: np.ndarray, fx: np.ndarray, i: int, top_k: int) -> np.ndarray:
    """Pull for particle i towards its NEAREST among the top_k
    best-performing particles (a small set of "attractive mates"),
    instead of the single swarm-wide best."""
    elite_idx = np.argsort(fx)[:top_k]
    elite_idx = elite_idx[elite_idx != i]
    if len(elite_idx) == 0:
        return np.zeros_like(x[i])
    dists = np.linalg.norm(x[elite_idx] - x[i], axis=1)
    nearest = elite_idx[np.argmin(dists)]
    return x[nearest] - x[i]


def ring_neighbors(n: int, i: int) -> tuple[int, int]:
    return (i - 1) % n, (i + 1) % n


def ring_lbest(pbest_x: np.ndarray, pbest_fx: np.ndarray, i: int) -> np.ndarray:
    left, right = ring_neighbors(len(pbest_x), i)
    candidates = [i, left, right]
    best = candidates[int(np.argmin(pbest_fx[candidates]))]
    return pbest_x[best]
