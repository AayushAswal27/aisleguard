from pathlib import Path
import random
import shutil

ROOT = Path("data/raw")

DATASETS = [
    "loco",
    "logistics",
    "vehicle",
    "forklift_1",
]

SAMPLES_PER_DATASET = 100

OUTPUT = Path("inspection_samples")

# Delete old inspection folder
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

OUTPUT.mkdir()

random.seed(42)

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

for dataset in DATASETS:

    dataset_root = ROOT / dataset

    images = []

    for split in ["train", "valid", "test"]:

        img_dir = dataset_root / split / "images"

        if not img_dir.exists():
            continue

        for ext in IMAGE_EXTS:
            images.extend(img_dir.glob(f"*{ext}"))

    images = list(images)

    if len(images) == 0:
        print(f"{dataset}: No images found.")
        continue

    sample_size = min(SAMPLES_PER_DATASET, len(images))
    sample = random.sample(images, sample_size)

    out_dir = OUTPUT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for img in sample:
        shutil.copy2(img, out_dir / img.name)

    print(f"{dataset}: copied {sample_size} images")

print("\n==============================")
print("Done!")
print(f"Samples saved in: {OUTPUT.resolve()}")