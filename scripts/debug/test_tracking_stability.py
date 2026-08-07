from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

MODEL_PATH = "models/yolo/best.pt"
VIDEOS = [f"evaluation_videos/cctv/Camera{n}.mp4" for n in range(1, 6)]

model = YOLO(MODEL_PATH)

for video in VIDEOS:
    print(f"\nProcessing {Path(video).name} ...", flush=True)

    results = model.track(
        source=video,
        tracker="botsort.yaml",
        persist=True,
        conf=0.4,
        verbose=False,
        stream=True,
    )

    frames = 0
    unique_ids = set()
    id_lifespans = defaultdict(int)   # frames each ID survives

    for r in results:
        frames += 1
        if r.boxes.id is None:
            continue
        for tid in r.boxes.id.cpu().numpy().astype(int):
            unique_ids.add(int(tid))
            id_lifespans[int(tid)] += 1

    long_tracks = sum(1 for v in id_lifespans.values() if v > 30)
    avg_life = sum(id_lifespans.values()) / max(len(id_lifespans), 1)

    print(f"  frames processed        : {frames}")
    print(f"  total unique IDs        : {len(unique_ids)}")
    print(f"  IDs surviving >30 frames: {long_tracks}")
    print(f"  avg track lifespan      : {avg_life:.0f} frames")

print("\nDone.")