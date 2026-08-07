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
        img = Image.open(img_path).convert("RGB")
        h = imagehash.phash(img)
        hashes[str(h)].append(img_path)

    except Exception:
        continue

duplicates = {
    h: files
    for h, files in hashes.items()
    if len(files) > 1
}

print("\n====================")
print("Duplicate Groups:", len(duplicates))

total = sum(len(v) for v in duplicates.values())

print("Duplicate Images:", total)

for h, files in list(duplicates.items())[:20]:

    print("\nHash:", h)

    for f in files:
        print(" ", f)