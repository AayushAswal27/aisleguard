# Roadmap

## Delivered (V1)

The full ten-stage pipeline is complete and demonstrable end to end:

- [x] YOLOv11 detection (2 classes, forklift mAP@50 0.94)
- [x] BoT-SORT tracking with persistent IDs
- [x] Homography → metric floor projection
- [x] Metric trajectory corpus
- [x] TTC / PET physics conflict metrics
- [x] Procedural conflict simulator (~400k labelled timesteps)
- [x] Bird's-eye velocity rasteriser
- [x] BRIN custom CNN (92.7% acc, 93% IMMINENT recall, beats baseline 93-vs-0)
- [x] Grad-CAM explainability
- [x] Two-tab Streamlit demo (interactive + live video)

## Future work

- **Per-camera calibration.** Accurate metric risk on arbitrary footage requires
  calibrating each camera's homography. The video demo currently uses an
  approximate frame-geometry scaling for uncalibrated cameras.
- **Detection robustness.** Fine-tune the detector on more real warehouse footage
  to reduce dropout on fast, motion-blurred forklifts.
- **Appearance-based re-identification.** Add a re-ID model (e.g. OSNet via BoxMOT)
  to eliminate track fragmentation under occlusion, replacing the demo's
  lightweight track-stitching.
- **Real-data fine-tuning.** BRIN is trained on simulated conflicts; fine-tuning on
  labelled real-world near-misses would close the sim-to-real gap.
- **Multi-timestep input.** Feed a short history of rasters instead of a single
  frame, so the network sees acceleration and intent, not just instantaneous motion.
- **Confidence calibration.** Temperature-scale the outputs so predicted
  probabilities are honest rather than saturated at 100%.