import numpy as np


# Floor bounds (metres) — matches your simulator aisle
X_MIN, X_MAX = 0.0, 3.5      # across aisle
Y_MIN, Y_MAX = 0.0, 20.0     # down aisle
GRID = 64                    # 64x64 cells


def _to_cell(x, y):
    """Convert a floor position in metres to a grid cell (col, row)."""
    cx = int((x - X_MIN) / (X_MAX - X_MIN) * GRID)
    cy = int((y - Y_MIN) / (Y_MAX - Y_MIN) * GRID)
    # clamp to valid range
    cx = min(max(cx, 0), GRID - 1)
    cy = min(max(cy, 0), GRID - 1)
    return cx, cy


def rasterise(snapshot):
    """
    Turn one timestep's agents into a (6, GRID, GRID) tensor.
    snapshot: list of dicts with keys cls, x, y, vx, vy
    Channels: [ped_occ, fork_occ, ped_vx, ped_vy, fork_vx, fork_vy]
    """
    raster = np.zeros((6, GRID, GRID), dtype=np.float32)

    for a in snapshot:
        cx, cy = _to_cell(a["x"], a["y"])
        if a["cls"] == "person":
            raster[0, cy, cx] = 1.0          # occupancy
            raster[2, cy, cx] = a["vx"]      # velocity x
            raster[3, cy, cx] = a["vy"]      # velocity y
        elif a["cls"] == "forklift":
            raster[1, cy, cx] = 1.0
            raster[4, cy, cx] = a["vx"]
            raster[5, cy, cx] = a["vy"]

    return raster