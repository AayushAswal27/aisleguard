from pathlib import Path

ROOT = Path("data/merged")

SPLITS = ["train", "valid", "test"]

bad = []

for split in SPLITS:

    label_dir = ROOT / split / "labels"

    for label_file in label_dir.glob("*.txt"):

        with open(label_file) as f:

            for line_no, line in enumerate(f, start=1):

                parts = line.strip().split()

                if len(parts) != 5:
                    bad.append(
                        (
                            split,
                            label_file,
                            line_no,
                            "Wrong number of values",
                            line.strip(),
                        )
                    )
                    continue

                cls = parts[0]

                try:
                    x, y, w, h = map(float, parts[1:])
                except Exception:
                    bad.append(
                        (
                            split,
                            label_file,
                            line_no,
                            "Cannot parse numbers",
                            line.strip(),
                        )
                    )
                    continue

                reason = []

                if not (0 <= x <= 1):
                    reason.append(f"x={x}")

                if not (0 <= y <= 1):
                    reason.append(f"y={y}")

                if not (0 < w <= 1):
                    reason.append(f"w={w}")

                if not (0 < h <= 1):
                    reason.append(f"h={h}")

                if reason:
                    bad.append(
                        (
                            split,
                            label_file,
                            line_no,
                            ", ".join(reason),
                            line.strip(),
                        )
                    )

print("=" * 70)
print(f"Found {len(bad)} bad boxes\n")

for item in bad[:100]:

    split, file, line_no, reason, content = item

    print(f"[{split}]")
    print(file)
    print(f"Line {line_no}")
    print(reason)
    print(content)
    print("-" * 70)

print("\nShowing first 100 bad boxes only.")