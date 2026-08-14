"""
Unit tests for the rasteriser (src/bev/rasteriser.py) and the BRIN model
(src/risk/predictor.py).

These check the data contract between the two: the rasteriser must emit exactly
the (6, 64, 64) tensor BRIN expects, and BRIN must consume it and emit three
class logits. They catch shape/channel regressions that would silently break
inference.

Run:  pytest tests/test_model.py
"""
import numpy as np
import torch
from src.bev.rasteriser import rasterise, GRID
from src.risk.predictor import BRIN


# ---------------------------------------------------------------- rasteriser
def test_raster_shape():
    """A snapshot rasterises to (6, GRID, GRID) float32."""
    snap = [{"cls": "person",   "x": 1.75, "y": 10.0, "vx": 0.2, "vy": 0.0},
            {"cls": "forklift", "x": 1.75, "y": 8.0,  "vx": 0.0, "vy": 3.0}]
    r = rasterise(snap)
    assert r.shape == (6, GRID, GRID)
    assert r.dtype == np.float32


def test_raster_channels():
    """Person writes to channels 0/2/3; forklift to 1/4/5."""
    snap = [{"cls": "person",   "x": 1.0, "y": 5.0, "vx": 0.5, "vy": -0.5},
            {"cls": "forklift", "x": 2.0, "y": 6.0, "vx": 1.0, "vy": 2.0}]
    r = rasterise(snap)
    # person occupancy present in channel 0, forklift in channel 1
    assert r[0].sum() == 1.0
    assert r[1].sum() == 1.0
    # velocity channels carry the signed values somewhere
    assert np.isclose(r[2].sum(), 0.5)    # person vx
    assert np.isclose(r[5].sum(), 2.0)    # forklift vy


def test_raster_empty_snapshot():
    """No agents -> all-zero raster of the right shape."""
    r = rasterise([])
    assert r.shape == (6, GRID, GRID)
    assert r.sum() == 0.0


# ---------------------------------------------------------------- BRIN
def test_brin_output_shape():
    """BRIN maps a batch of rasters to (B, 3) logits."""
    model = BRIN().eval()
    x = torch.randn(4, 6, GRID, GRID)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 3)


def test_brin_consumes_rasteriser_output():
    """The rasteriser's output feeds straight into BRIN and yields 3 logits."""
    snap = [{"cls": "person",   "x": 1.75, "y": 12.0, "vx": 0.0, "vy": -1.4},
            {"cls": "forklift", "x": 1.75, "y": 6.0,  "vx": 0.0, "vy": 3.0}]
    r = rasterise(snap)
    model = BRIN().eval()
    with torch.no_grad():
        out = model(torch.tensor(r[None]))     # add batch dim
    assert out.shape == (1, 3)
    # softmax is a valid probability distribution
    probs = torch.softmax(out, dim=1)[0]
    assert abs(float(probs.sum()) - 1.0) < 1e-5