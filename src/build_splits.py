from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def keep_labeled(images: list[Path], labels_dir: Path) -> list[Path]:
    labeled = []
    for image in images:
        label_path = labels_dir / f"{image.stem}.txt"
        if label_path.exists():
            labeled.append(image)
    return labeled


def write_split(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"{p.as_posix()}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test split files for YOLO.")
    parser.add_argument("--images-dir", type=str, default="data/labeled/images")
    parser.add_argument("--labels-dir", type=str, default="data/labeled/labels")
    parser.add_argument("--out-dir", type=str, default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)

    images = collect_images(images_dir)
    images = keep_labeled(images, labels_dir)
    if not images:
        raise ValueError("no labeled images found")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    total = len(images)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    train_paths = images[:train_end]
    val_paths = images[train_end:val_end]
    test_paths = images[val_end:]

    write_split(train_paths, out_dir / "train.txt")
    write_split(val_paths, out_dir / "val.txt")
    write_split(test_paths, out_dir / "test.txt")

    print(f"Total images: {total}")
    print(f"Train/Val/Test: {len(train_paths)}/{len(val_paths)}/{len(test_paths)}")
    print(f"Wrote split files to: {out_dir}")


if __name__ == "__main__":
    main()

