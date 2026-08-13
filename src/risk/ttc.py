"""
Conflict metrics: Time-To-Collision (TTC) and Post-Encroachment-Time (PET).

These physics-based functions define what a "conflict" is. They are used to
LABEL the simulated training data (Phase 5-6); they are never called at
inference — BRIN predicts from the raster alone. TTC assumes constant velocity
and solves for when two circular agents' separation first drops below the sum
of their radii; PET measures the smallest time gap between two agents occupying
the same floor cell.
"""
import numpy as np


def time_to_collision(pos_a, vel_a, pos_b, vel_b, radius_a=0.4, radius_b=1.5):
    """
    TTC between two agents assuming constant velocity.
    pos/vel are (x, y) in metres and m/s.
    Returns seconds until collision, or np.inf if they never collide.
    """
    dp = np.array(pos_b, dtype=float) - np.array(pos_a, dtype=float)   # relative position
    dv = np.array(vel_b, dtype=float) - np.array(vel_a, dtype=float)   # relative velocity
    R = radius_a + radius_b

    # already overlapping -> collision now
    if np.linalg.norm(dp) <= R:
        return 0.0

    a = dv @ dv                     # |Δv|²
    b = 2 * (dp @ dv)               # 2(Δp·Δv)
    c = dp @ dp - R * R             # |Δp|² - R²

    if a < 1e-9:                    # no relative motion -> never closes
        return np.inf

    disc = b * b - 4 * a * c
    if disc < 0:                    # no real root -> paths never intersect within R
        return np.inf

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # smallest positive root
    candidates = [t for t in (t1, t2) if t >= 0]
    return min(candidates) if candidates else np.inf


def post_encroachment_time(track_a, track_b, cell_size=0.5):
    """
    PET: smallest time gap between two agents occupying the same floor cell.
    track_a, track_b are lists of (t, x, y).
    Returns seconds, or np.inf if their paths never share a cell.
    """
    def cells(track):
        d = {}
        for t, x, y in track:
            key = (round(x / cell_size), round(y / cell_size))
            d.setdefault(key, []).append(t)
        return d

    ca, cb = cells(track_a), cells(track_b)
    best = np.inf
    for key in set(ca) & set(cb):          # shared cells
        for ta in ca[key]:
            for tb in cb[key]:
                best = min(best, abs(ta - tb))
    return best