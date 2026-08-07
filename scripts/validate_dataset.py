from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm import tqdm

ROOT = Path("data/merged")

SPLITS = ["train", "valid", "test"]

VALID_CLASSES = {0, 1}

image_extensions = [".jpg", ".jpeg", ".png"]

stats = Counter()

print("=" * 60)
print("AISLEGUARD DATASET VALIDATOR")
print("=" * 60)

for split in SPLITS:

    print(f"\nChecking {split}...")

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = {}

    for ext in image_extensions:
        for img in image_dir.glob(f"*{ext}"):
            images[img.stem] = img

    labels = {x.stem: x for x in label_dir.glob("*.txt")}

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    missing_labels = set(images) - set(labels)
    missing_images = set(labels) - set(images)

    print(f"Missing labels : {len(missing_labels)}")
    print(f"Missing images : {len(missing_images)}")

    stats["images"] += len(images)

    corrupted = 0
    empty = 0
    bad_boxes = 0
    invalid_classes = 0

    for stem, label_file in tqdm(labels.items()):

        img_path = images.get(stem)

        if img_path is None:
            continue

        try:
            Image.open(img_path).verify()
        except Exception:
            corrupted += 1
            continue

        lines = label_file.read_text().strip().splitlines()

        if len(lines) == 0:
            empty += 1
            continue

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                bad_boxes += 1
                continue

            cls = int(parts[0])

            if cls not in VALID_CLASSES:
                invalid_classes += 1

            x, y, w, h = map(float, parts[1:])

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                bad_boxes += 1

            stats[f"class_{cls}"] += 1

    print(f"Corrupted images : {corrupted}")
    print(f"Empty labels     : {empty}")
    print(f"Bad boxes        : {bad_boxes}")
    print(f"Invalid classes  : {invalid_classes}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total images : {stats['images']}")
print(f"Persons      : {stats['class_0']}")
print(f"Forklifts    : {stats['class_1']}")

ratio = (
    stats["class_0"] / stats["class_1"]
    if stats["class_1"] > 0
    else 0
)

print(f"Person/Forklift ratio : {ratio:.2f}:1")

print("\nValidation Complete.")