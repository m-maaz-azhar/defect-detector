from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_ratio_sum(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")


def normalize(value: float, denom: float) -> float:
    if denom <= 0:
        raise ValueError("normalization denominator must be > 0")
    return clamp(value / denom, 0.0, 1.0)


def resolve_image_path(images_dir: Path, filename: str, stem: str) -> Path | None:
    direct = images_dir / filename
    if direct.exists():
        return direct

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_voc(xml_path: Path) -> tuple[str, int, int, list[tuple[str, tuple[float, float, float, float]]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = (root.findtext("filename") or "").strip()
    if not filename:
        filename = f"{xml_path.stem}.jpg"

    width = int(float(root.findtext("./size/width", "0")))
    height = int(float(root.findtext("./size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in annotation: {xml_path}")

    objects: list[tuple[str, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        bbox = obj.find("bndbox")
        if not class_name or bbox is None:
            continue

        try:
            xmin = float((bbox.findtext("xmin") or "0").strip())
            ymin = float((bbox.findtext("ymin") or "0").strip())
            xmax = float((bbox.findtext("xmax") or "0").strip())
            ymax = float((bbox.findtext("ymax") or "0").strip())
        except ValueError:
            continue

        x1 = clamp(min(xmin, xmax), 0.0, float(width))
        y1 = clamp(min(ymin, ymax), 0.0, float(height))
        x2 = clamp(max(xmin, xmax), 0.0, float(width))
        y2 = clamp(max(ymin, ymax), 0.0, float(height))
        if x2 <= x1 or y2 <= y1:
            continue

        objects.append((class_name, (x1, y1, x2, y2)))

    return filename, width, height, objects


def bbox_to_yolo_seg_line(
    class_id: int,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = bbox
    points = (
        normalize(x1, width),
        normalize(y1, height),
        normalize(x2, width),
        normalize(y1, height),
        normalize(x2, width),
        normalize(y2, height),
        normalize(x1, width),
        normalize(y2, height),
    )
    points_str = " ".join(f"{value:.6f}" for value in points)
    return f"{class_id} {points_str}"


def to_posix_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def write_text_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        if lines:
            handle.write("\n")


def write_splits(
    image_paths: list[Path],
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[int, int, int]:
    safe_ratio_sum(train_ratio, val_ratio, test_ratio)
    if not image_paths:
        raise ValueError("cannot build splits without prepared images")

    rng = random.Random(seed)
    shuffled = image_paths.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_paths = shuffled[:train_end]
    val_paths = shuffled[train_end:val_end]
    test_paths = shuffled[val_end:]

    write_text_file(out_dir / "train.txt", [to_posix_relative(path) for path in train_paths])
    write_text_file(out_dir / "val.txt", [to_posix_relative(path) for path in val_paths])
    write_text_file(out_dir / "test.txt", [to_posix_relative(path) for path in test_paths])
    return len(train_paths), len(val_paths), len(test_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VOC XML boxes into YOLO-seg labels (rectangle polygons) and optionally write splits."
    )
    parser.add_argument("--annotations-dir", type=str, default="data/annotations")
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--out-images-dir", type=str, default="data/labeled/images")
    parser.add_argument("--out-labels-dir", type=str, default="data/labeled/labels")
    parser.add_argument("--single-class-name", type=str, default="defect")
    parser.add_argument("--use-source-classes", action="store_true")
    parser.add_argument("--include-negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safe_ratio_sum(args.train_ratio, args.val_ratio, args.test_ratio)

    annotations_dir = Path(args.annotations_dir)
    images_dir = Path(args.images_dir)
    out_images_dir = Path(args.out_images_dir)
    out_labels_dir = Path(args.out_labels_dir)
    splits_dir = Path(args.splits_dir)

    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations directory not found: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"no XML annotations found in {annotations_dir}")

    class_to_id: dict[str, int] = {}
    if not args.use_source_classes:
        class_to_id[args.single_class_name] = 0

    prepared_images: list[Path] = []
    missing_images = 0
    converted_objects = 0
    skipped_empty = 0

    for xml_path in xml_files:
        filename, width, height, objects = parse_voc(xml_path)
        image_path = resolve_image_path(images_dir, filename, xml_path.stem)
        if image_path is None:
            missing_images += 1
            continue

        yolo_lines: list[str] = []
        for source_name, bbox in objects:
            class_name = source_name if args.use_source_classes else args.single_class_name
            if class_name not in class_to_id:
                class_to_id[class_name] = len(class_to_id)
            class_id = class_to_id[class_name]
            yolo_lines.append(bbox_to_yolo_seg_line(class_id, bbox, width, height))

        if not yolo_lines and not args.include_negatives:
            skipped_empty += 1
            continue

        out_image_path = out_images_dir / image_path.name
        out_label_path = out_labels_dir / f"{Path(filename).stem}.txt"
        prepared_images.append(out_image_path)

        if not args.dry_run:
            out_image_path.parent.mkdir(parents=True, exist_ok=True)
            out_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, out_image_path)
            write_text_file(out_label_path, yolo_lines)

        converted_objects += len(yolo_lines)

    if not prepared_images:
        raise ValueError("no samples were prepared; check annotations, labels, and --include-negatives setting")

    split_counts = None
    if args.write_splits and not args.dry_run:
        split_counts = write_splits(
            image_paths=prepared_images,
            out_dir=splits_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    print(f"Annotations processed: {len(xml_files)}")
    print(f"Samples prepared: {len(prepared_images)}")
    print(f"Objects converted: {converted_objects}")
    print(f"Missing images: {missing_images}")
    if skipped_empty:
        print(f"Skipped empty annotations: {skipped_empty}")
    if args.use_source_classes:
        ordered = ", ".join(
            f"{class_name}:{class_id}" for class_name, class_id in sorted(class_to_id.items(), key=lambda item: item[1])
        )
        print(f"Class mapping: {ordered}")
    else:
        print(f"Class mapping: {args.single_class_name}:0")
    if split_counts is not None:
        train_count, val_count, test_count = split_counts
        print(f"Split counts (train/val/test): {train_count}/{val_count}/{test_count}")
    if args.dry_run:
        print("Dry run enabled: no files were written.")


if __name__ == "__main__":
    main()
