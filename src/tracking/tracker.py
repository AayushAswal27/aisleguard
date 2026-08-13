"""
Detection + tracking runner (Phase 2).

Runs YOLOv11 with BoT-SORT tracking over a fixed-camera video and writes a
trajectory CSV: one row per tracked detection with frame index, persistent
track ID, class, confidence, and the bottom-centre pixel (the ground-contact
point that Phase 3's homography projects to metric floor coordinates).
"""
from ultralytics import YOLO
import cv2
import csv
from pathlib import Path


MODEL_PATH = "models/yolo/best.pt"
VIDEO_PATH = "evaluation_videos/cctv/Camera1.mp4"
OUTPUT_CSV = "outputs/trajectories/camera1_tracks.csv"


def main():
    """Track a video end-to-end and dump per-detection rows to OUTPUT_CSV."""
    model = YOLO(MODEL_PATH)

    results = model.track(
        source=VIDEO_PATH,
        tracker="botsort.yaml",
        persist=True,
        save=True,
        show=True,
        conf=0.4,
        verbose=False,
    )

    rows = []   # collect all trajectory data here

    for frame_idx, result in enumerate(results):

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names

        print(f"\nFrame {frame_idx}")

        for box, track_id, cls, conf in zip(boxes, ids, classes, confidences):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = y2   # bottom-center = ground-contact point

            rows.append([frame_idx, track_id, names[cls], round(conf, 3),
                         round(cx, 1), round(cy, 1)])   # save each detection

            print(
                f"ID={track_id:3d} | {names[cls]:10s} | "
                f"Conf={conf:.2f} | BottomCenter=({cx:.1f}, {cy:.1f})"
            )

    # after all frames: write the CSV
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class", "conf", "cx", "cy"])
        w.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()