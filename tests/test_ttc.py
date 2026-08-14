"""
Unit tests for the TTC / PET conflict metrics (src/risk/ttc.py).

These lock down the physics that labels the training data: if TTC/PET drift,
every downstream label is wrong. Cases cover a head-on approach, diverging and
parallel motion (no collision), an already-overlapping pair, and a PET crossing.

Run:  pytest tests/test_ttc.py
"""
import numpy as np
from src.risk.ttc import time_to_collision, post_encroachment_time


def test_head_on_collision():
    """Two agents closing head-on should collide at a finite, positive time."""
    # forklift at x=0 moving +x at 1 m/s; person at x=10 moving -x at 1 m/s
    # radii 1.5 + 0.4 = 1.9; gap closes at 2 m/s; contact when separation = 1.9
    ttc = time_to_collision((0, 0), (1, 0), (10, 0), (-1, 0),
                            radius_a=1.5, radius_b=0.4)
    assert np.isfinite(ttc)
    assert abs(ttc - 4.05) < 0.05      # (10 - 1.9) / 2 = 4.05 s


def test_diverging_never_collides():
    """Agents moving apart never collide -> inf."""
    ttc = time_to_collision((0, 0), (-1, 0), (10, 0), (1, 0),
                            radius_a=1.5, radius_b=0.4)
    assert ttc == np.inf


def test_parallel_never_collides():
    """Agents moving in parallel with no closing component -> inf."""
    ttc = time_to_collision((0, 0), (0, 1), (5, 0), (0, 1),
                            radius_a=1.5, radius_b=0.4)
    assert ttc == np.inf


def test_already_overlapping():
    """Agents already within combined radius -> collision now (0.0)."""
    ttc = time_to_collision((0, 0), (0, 0), (0.5, 0), (0, 0),
                            radius_a=1.5, radius_b=0.4)
    assert ttc == 0.0


def test_no_relative_motion():
    """Stationary, non-overlapping agents never collide -> inf."""
    ttc = time_to_collision((0, 0), (0, 0), (10, 0), (0, 0),
                            radius_a=1.5, radius_b=0.4)
    assert ttc == np.inf


def test_pet_crossing():
    """Two tracks passing through the same cell 2 s apart -> PET = 2.0."""
    # both visit floor cell near (5, 5); A at t=1.0, B at t=3.0
    track_a = [(0.0, 0, 5), (1.0, 5, 5), (2.0, 10, 5)]
    track_b = [(2.0, 5, 0), (3.0, 5, 5), (4.0, 5, 10)]
    pet = post_encroachment_time(track_a, track_b, cell_size=0.5)
    assert abs(pet - 2.0) < 1e-6


def test_pet_no_shared_cell():
    """Tracks that never share a cell -> inf."""
    track_a = [(0.0, 0, 0), (1.0, 1, 0)]
    track_b = [(0.0, 10, 10), (1.0, 11, 10)]
    pet = post_encroachment_time(track_a, track_b, cell_size=0.5)
    assert pet == np.inf