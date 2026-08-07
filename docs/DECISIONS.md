# Architecture Decision Records

## ADR-001: Two classes only (person, forklift) for V1
Dropped PPE/pallet/rack classes despite collecting them. The risk model
consumes only person + forklift position and velocity — PPE never enters
the BEV raster. Partially-annotated extra classes would have hurt the
detector without adding downstream value. Additional classes deferred to V2.

## ADR-002: Class remapping by name, not index
Merge script maps source classes by name read from each data.yaml, not by
hardcoded index. An earlier index-based approach silently miscounted a
source's `small_load_carrier` (index 3) as `person` (index 3). Name-based
mapping makes adding new sources safe.

## ADR-003: BoT-SORT over ByteTrack
Warehouse workers wear near-identical hi-vis; motion-only tracking swaps
their IDs when paths cross. An ID switch fabricates a false velocity vector,
which would produce a false conflict prediction downstream. BoT-SORT's
appearance embedding reduces this.

## ADR-004: Fixed cameras only
Homography (Phase 3) is valid only for a single fixed viewpoint. Moving-camera
footage (e.g. handheld DVIDS clips) breaks metric projection and is excluded
from the target domain. Confirmed by testing: moving-camera clips showed
severe ID churn and box-edge clipping; fixed-camera (TalTech) footage tracked
cleanly.

## ADR-005: Perceptual-hash dedup before train/val/test split
Roboflow datasets are heavily re-forked; the same image appears across
sources. Deduplicating before splitting prevents the same scene landing in
both train and test, which would inflate reported mAP.
