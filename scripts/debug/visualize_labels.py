from pathlib import Path
import random
import cv2

ROOT = Path("data/raw")

DATASETS = [
    "loco",
    "forklift_1",
    "vehicle",
    "logistics",
]

OUTPUT = Path("label_inspection")

if OUTPUT.exists():
    import shutil
    shutil.rmtree(OUTPUT)

OUTPUT.mkdir()

random.seed(42)

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


def find_image(image_dir, stem):
    for ext in IMAGE_EXTS:
        p = image_dir / (stem + ext)
        if p.exists():
            return p
    return None


for dataset in DATASETS:

    out_dir = OUTPUT / dataset
    out_dir.mkdir()

    images_saved = 0

    train_labels = ROOT / dataset / "train" / "labels"

    if not train_labels.exists():
        continue

    label_files = list(train_labels.glob("*.txt"))
    random.shuffle(label_files)

    for label_file in label_files:

        if images_saved >= 25:
            break

        image = find_image(ROOT / dataset / "train" / "images", label_file.stem)

        if image is None:
            continue

        img = cv2.imread(str(image))

        if img is None:
            continue

        h, w = img.shape[:2]

        with open(label_file) as f:
            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    str(cls),
                    (x1, max(20, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        cv2.imwrite(str(out_dir / image.name), img)
        images_saved += 1

print("\nDone!")
print("Open folder: label_inspection/")