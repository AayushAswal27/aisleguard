from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm

ROOT = Path("data/merged")
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

hashes = defaultdict(list)

images = []

for ext in IMAGE_EXTS:
    images.extend(ROOT.rglob(f"*{ext}"))

print(f"Scanning {len(images)} images...\n")

for img_path in tqdm(images):
    try:
        h = str(imagehash.phash(Image.open(img_path)))
        hashes[h].append(img_path)
    except Exception:
        pass


deleted = 0

for group in hashes.values():

    if len(group) <= 1:
        continue

    # Prefer keeping TRAIN image
    group = sorted(
        group,
        key=lambda p: (
            "train" not in str(p),
            "valid" not in str(p),
            "test" not in str(p),
        ),
    )

    keep = group[0]

    for img in group[1:]:

        label = (
            Path(str(img).replace("/images/", "/labels/"))
            .with_suffix(".txt")
        )

        try:
            img.unlink()
            deleted += 1
        except FileNotFoundError:
            pass

        if label.exists():
            label.unlink()

print(f"\nDeleted {deleted} duplicate images.")
print("Done.")