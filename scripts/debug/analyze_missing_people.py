from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------

ROOT = Path("data/raw/loco")

MODEL = "models/yolo11x.pt"

CONF = 0.45

model = YOLO(MODEL)

splits = ["train", "valid", "test"]

images_scanned = 0
existing_people = 0
detected_people = 0

print("=" * 60)
print("ANALYZING LOCO DATASET")
print("=" * 60)

for split in splits:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = list(image_dir.glob("*"))

    print(f"\n{split}: {len(images)} images")

    for image in tqdm(images):

        images_scanned += 1

        label_file = label_dir / (image.stem + ".txt")

        # Count existing person labels
        if label_file.exists():

            with open(label_file) as f:

                for line in f:

                    if line.strip():

                        cls = int(line.split()[0])

                        if cls == 3:          # LOCO person class
                            existing_people += 1

        # Detect persons with YOLO11x
        results = model.predict(
            source=str(image),
            classes=[0],      # COCO person
            conf=CONF,
            verbose=False,
        )

        detected_people += len(results[0].boxes)

print("\n")
print("=" * 60)
print("RESULT")
print("=" * 60)

print(f"Images scanned          : {images_scanned}")
print(f"Existing person labels  : {existing_people}")
print(f"YOLO detected persons   : {detected_people}")
print(f"Potential extra persons : {detected_people-existing_people}")