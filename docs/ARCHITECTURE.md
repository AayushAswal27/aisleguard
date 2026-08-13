# Architecture

AisleGuard is a ten-stage pipeline that turns a single fixed-camera video into a
forward-looking forklift–pedestrian conflict prediction. This document walks the
data path end to end and records the shape of the data at each stage.

## Overview

```
Camera frame
  → [1] Detection        person / forklift boxes
  → [2] Tracking         persistent IDs + motion
  → [3] Homography       pixel → metric floor coordinates
  → [4] Trajectory       position (m) + velocity (m/s) per track
  → [5] TTC / PET         physics-based conflict labels
  → [6] Simulator        400k labelled conflict scenarios
  → [7] Rasteriser       (6, 64, 64) bird's-eye velocity tensors
  → [8] BRIN             SAFE / CAUTION / IMMINENT
  → [9] Grad-CAM         explainability overlay
  → [10] Demo            interactive + live-video interface
```

## Stage detail

### [1] Detection
YOLOv11-s, two classes (`person`, `forklift`), trained on a merged, deduplicated
23,778-image dataset (see `DATASET.md`). Output per frame: bounding boxes with
class and confidence. Forklift mAP@50 = 0.94.

### [2] Tracking
BoT-SORT (Kalman motion model + appearance embedding) assigns a persistent ID to
each detection across frames. Persistent IDs are required to compute velocity —
without a stable ID there is no way to measure how far an object moved between
frames. BoT-SORT is preferred over ByteTrack because an ID switch fabricates a
false velocity vector and therefore a false conflict.

### [3] Homography
A 3×3 homography maps floor pixels to metric floor coordinates. Calibrated from
four known floor points; validated to 0 cm reprojection on the calibration points
and to a sensible (1.06 m, 9.74 m) on a real tracked forklift. The bottom-centre
of each box is used as the ground-contact point, since only that point lies on
the floor plane.

### [4] Trajectory corpus
Per track: metric position and finite-difference velocity, with a class-lock
(majority vote per track) and 5-frame rolling-mean velocity smoothing to remove
tracker jitter. Forklift speeds (0.66 m/s mean, 5.0 m/s max) fall in the expected
warehouse range, validating the metric scale.

### [5] TTC / PET labelling
Time-to-collision (constant-velocity quadratic on relative position/velocity with
agent radii) and post-encroachment-time define what a conflict is. These physics
functions LABEL data; they are never called at inference.

### [6] Simulator
A procedural NumPy simulator generates forklift–pedestrian scenarios (real
near-misses cannot be safely staged). Each timestep is labelled by gating on both
TTC and distance. 2,000 scenarios → ~400k labelled timesteps
(SAFE 61% / IMMINENT 28% / CAUTION 11%).

### [7] Rasteriser
Each timestep is rendered to a (6, 64, 64) tensor over the aisle floor. Channels:
`[ped_occ, fork_occ, ped_vx, ped_vy, fork_vx, fork_vy]` — velocity is encoded as
signed metres/second directly in the raster so a CNN can read motion spatially.

### [8] BRIN
A custom residual CNN (~491k parameters) trained from scratch — the modality has
no pretrained weights. Input (B, 6, 64, 64) → logits (B, 3). See `TRAINING.md`.

### [9] Grad-CAM
Gradient-weighted class activation mapping on the final conv layer confirms the
network attends to the collision zone between the two agents. Because the input is
in metric floor-space, the heatmap overlays directly onto the real floor.

### [10] Demo
A two-tab Streamlit app: an interactive risk-model sandbox, and a full-pipeline
video mode that runs detection → tracking → projection → BRIN on uploaded footage.

## Design notes

- **Metric floor-space, not pixels.** Every downstream stage reasons in metres, so
  risk reflects real proximity rather than perspective-distorted pixel distance.
- **Physics labels, network predicts.** TTC/PET generate labels; BRIN predicts from
  the raster alone and generalises beyond a constant-velocity formula.
- **Trained from scratch by necessity.** Velocity rasters have no pretrained weights;
  transfer learning is inapplicable, not merely skipped.