from pathlib import Path
import random
import cv2

ROOT = Path("data/merged/train")

images = list(ROOT.joinpath("images").glob("loco_*"))

print(f"Found {len(images)} LOCO images")

random.seed(42)
samples = random.sample(images, min(25, len(images)))

OUT = Path("check_loco")
OUT.mkdir(exist_ok=True)

COLORS = {
    0: (0, 255, 0),      # Person = Green
    1: (0, 0, 255),      # Forklift = Red
}

NAMES = {
    0: "PERSON",
    1: "FORKLIFT",
}

for image_path in samples:

    img = cv2.imread(str(image_path))

    h, w = img.shape[:2]

    label_path = ROOT / "labels" / (image_path.stem + ".txt")

    if not label_path.exists():
        continue

    with open(label_path) as f:

        for line in f:

            cls, xc, yc, bw, bh = map(float, line.split())

            cls = int(cls)

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)

            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cls], 2)

            cv2.putText(
                img,
                NAMES[cls],
                (x1, max(25, y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLORS[cls],
                2,
            )

    cv2.imwrite(str(OUT / image_path.name), img)

print("\nDone!")
print(f"Saved to: {OUT.resolve()}")