# AisleGuard

**Predictive forklift–pedestrian conflict detection for warehouse floors.**

Forklift "struck-by" incidents are among the most expensive recurring safety failures in warehousing — but the near-misses that predict them are almost never recorded. AisleGuard is a computer-vision system that detects and tracks forklifts and pedestrians from a single fixed camera, projects them into metric floor coordinates, and forecasts conflict risk *before* it happens — giving safety systems seconds of lead time instead of a post-incident report.

> **Status: in active development.** Phases 1–2 (detection + tracking) are complete and working. Homography, risk modelling, and the app layer are in progress. See the roadmap below.

---

## Why this is not a proximity alarm

Most warehouse-safety CV projects measure pixel distance between boxes and fire an alert when they get close. Two problems: pixel distance isn't physical distance (perspective makes a forklift 15 m away look adjacent to a foreground worker), and an alert with zero lead time has zero value. AisleGuard reframes the problem as **spatiotemporal forecasting in metric ground coordinates** — it answers "will these two collide in the next 2 seconds," not "are these two boxes overlapping right now."

---

## Pipeline

```
Fixed camera
   ↓
YOLOv11  — detect person + forklift
   ↓
BoT-SORT — persistent track IDs + velocity
   ↓
Homography — project foot positions to metric floor coordinates   [in progress]
   ↓
TTC / PET — geometric conflict labelling                          [in progress]
   ↓
Custom CNN (BRIN) — risk forecast from bird's-eye motion rasters  [planned]
   ↓
Streamlit app — upload video, get risk overlays + alert log       [planned]
```

---

## Current results (Phases 1–2)

**Detection** — YOLOv11s trained on a self-merged, deduplicated 23,064-image dataset (person + forklift), evaluated on a held-out test set:

| Class | mAP@50 | mAP@50-95 |
|---|---|---|
| forklift | 0.937 | 0.731 |
| person | 0.869 | 0.623 |
| **overall** | **0.903** | **0.677** |

Inference: ~5.4 ms/frame (≈185 FPS on a T4). The ~1.5% val→test gap confirms no overfitting or split leakage.

**Tracking** — BoT-SORT produces stable persistent IDs on fixed-camera footage, with smooth per-frame trajectories from which velocity is derived. Ground-contact points (bottom-centre of each box) are extracted per frame as the input to metric projection.

---

## Dataset engineering

The training set was built, not downloaded. Five public warehouse/PPE sources were merged into a unified 2-class schema, then cleaned through a custom pipeline:

- Class remapping by name (not index) across heterogeneous source schemas
- Perceptual-hash deduplication — removed 8,767 near-duplicate images
- Polygon→bounding-box conversion for segmentation-annotated sources
- Full label validation (missing/corrupt/empty/out-of-range checks)
- Cross-split leakage verification (zero shared images across train/val/test)

Final: 23,064 images · 65,505 person + 7,650 forklift instances · verified clean.

A separate evaluation set of fixed-camera warehouse videos (TalTech CCTV, DVIDS, others) was collected for pipeline testing and is not used in training.

---

## Roadmap

| Phase | Status |
|---|---|
| 1. YOLOv11 detection | ✅ Complete |
| 2. BoT-SORT tracking + trajectory extraction | ✅ Complete |
| 3. Camera calibration + homography (metric floor projection) | 🔨 In progress |
| 4. Trajectory corpus assembly | ⏳ Planned |
| 5. TTC/PET geometric conflict labelling | ⏳ Planned |
| 6. Procedural conflict simulator | ⏳ Planned |
| 7. Bird's-eye-view rasteriser | ⏳ Planned |
| 8. BRIN — custom CNN risk model (from scratch) | ⏳ Planned |
| 9. Explainability (Grad-CAM, ablations) | ⏳ Planned |
| 10. Streamlit application | ⏳ Planned |

---

## Stack

YOLOv11 (Ultralytics) · BoT-SORT · OpenCV · PyTorch · NumPy · pandas · Streamlit

## Repository layout

```
src/
  detection/    YOLO wrapper
  tracking/     BoT-SORT wrapper + trajectory extraction
  calibration/  homography (in progress)
  bev/          BEV rasteriser (planned)
  risk/         TTC/PET + CNN risk model (planned)
scripts/        dataset engineering pipeline
notebooks/      per-phase development notebooks
docs/           architecture and decision records
```

---

*Built by [Aayush Aswal](https://github.com/AayushAswal27).*