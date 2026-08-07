from pathlib import Path
import shutil

ROOT = Path("data/merged")

SPLITS = ["train", "valid", "test"]

converted = 0
already_ok = 0
invalid = 0

print("=" * 60)
print("Converting Polygon Labels → YOLO Bounding Boxes")
print("=" * 60)

for split in SPLITS:

    label_dir = ROOT / split / "labels"

    backup_dir = ROOT / split / "labels_backup"

    if not backup_dir.exists():
        shutil.copytree(label_dir, backup_dir)

    for label_file in label_dir.glob("*.txt"):

        new_lines = []

        changed = False

        with open(label_file) as f:

            for line in f:

                parts = line.strip().split()

                # Already YOLO Detection format
                if len(parts) == 5:
                    new_lines.append(line.strip())
                    already_ok += 1
                    continue

                # Polygon format
                if len(parts) >= 9 and len(parts) % 2 == 1:

                    cls = parts[0]

                    coords = list(map(float, parts[1:]))

                    xs = coords[0::2]
                    ys = coords[1::2]

                    xmin = min(xs)
                    xmax = max(xs)

                    ymin = min(ys)
                    ymax = max(ys)

                    xc = (xmin + xmax) / 2
                    yc = (ymin + ymax) / 2

                    w = xmax - xmin
                    h = ymax - ymin

                    new_lines.append(
                        f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                    )

                    converted += 1
                    changed = True

                else:

                    invalid += 1

        if changed:

            with open(label_file, "w") as f:

                for line in new_lines:
                    f.write(line + "\n")

print("\n")
print("=" * 60)
print("Finished")
print("=" * 60)

print(f"Converted polygons : {converted}")
print(f"Already detection  : {already_ok}")
print(f"Invalid labels     : {invalid}")
print("\nBackups saved as labels_backup/")