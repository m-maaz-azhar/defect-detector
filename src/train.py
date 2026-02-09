from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


DEFAULT_CONFIG: dict = {
    "model": "yolov8m-seg.pt",
    "data": "configs/dataset.yaml",
    "project": "runs/segment",
    "name": "defect_poc",
    "epochs": 80,
    "imgsz": 960,
    "batch": 8,
    "device": "cpu",
    "workers": 4,
    "patience": 20,
    "pretrained": True,
    "optimizer": "auto",
    "seed": 42,
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.4,
    "degrees": 4.0,
    "translate": 0.08,
    "scale": 0.2,
    "perspective": 0.001,
    "fliplr": 0.5,
    "mosaic": 0.3,
    "copy_paste": 0.1,
    "export_onnx": True,
    "best_weights_out": "models/weights/best.pt",
    "onnx_out": "models/weights/model.onnx",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

TRAIN_ARG_KEYS = {
    "data",
    "epochs",
    "imgsz",
    "batch",
    "device",
    "workers",
    "patience",
    "pretrained",
    "optimizer",
    "seed",
    "project",
    "name",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "perspective",
    "fliplr",
    "mosaic",
    "copy_paste",
}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    config = {**DEFAULT_CONFIG, **user_cfg}
    return config


def copy_artifact(src: Path | None, dst: Path) -> None:
    if src is None or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _resolve_existing_path(raw_path: str, dataset_yaml_path: Path, dataset_root: str | None) -> Path:
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj

    candidates = []
    if dataset_root:
        root_obj = Path(dataset_root)
        if root_obj.is_absolute():
            candidates.append(root_obj / path_obj)
        else:
            candidates.append((Path.cwd() / root_obj / path_obj))
            candidates.append((dataset_yaml_path.parent / root_obj / path_obj))

    candidates.append(Path.cwd() / path_obj)
    candidates.append(dataset_yaml_path.parent / path_obj)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_split_line_path(line: str, split_file: Path) -> Path:
    line_path = Path(line)
    if line_path.is_absolute():
        return line_path

    candidates = [
        Path.cwd() / line_path,
        split_file.parent / line_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _validate_split_file(split_file: Path, split_name: str) -> None:
    if not split_file.exists():
        raise FileNotFoundError(f"{split_name} split file does not exist: {split_file}")
    if split_file.is_dir():
        image_count = sum(
            1
            for image_path in split_file.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count == 0:
            raise ValueError(f"{split_name} image directory is empty: {split_file}")
        return

    with open(split_file, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"{split_name} split file is empty: {split_file}")

    missing = []
    for line in lines:
        image_path = _resolve_split_line_path(line, split_file)
        if not image_path.exists():
            missing.append(str(image_path))

    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{split_name} split has missing image paths ({len(missing)} total). "
            f"Examples: {preview}"
        )


def validate_dataset_config(dataset_yaml: str | Path) -> None:
    dataset_yaml_path = Path(dataset_yaml)
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"dataset config not found: {dataset_yaml_path}")

    with open(dataset_yaml_path, "r", encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle) or {}

    required = ("train", "val", "names")
    for key in required:
        if key not in dataset_cfg:
            raise ValueError(f"dataset config missing required key: {key}")

    dataset_root = dataset_cfg.get("path")
    train_path = _resolve_existing_path(str(dataset_cfg["train"]), dataset_yaml_path, dataset_root)
    val_path = _resolve_existing_path(str(dataset_cfg["val"]), dataset_yaml_path, dataset_root)
    _validate_split_file(train_path, "train")
    _validate_split_file(val_path, "val")

    if "test" in dataset_cfg and dataset_cfg["test"] is not None:
        test_path = _resolve_existing_path(str(dataset_cfg["test"]), dataset_yaml_path, dataset_root)
        if test_path.exists():
            _validate_split_file(test_path, "test")

    names = dataset_cfg.get("names")
    if isinstance(names, dict) and not names:
        raise ValueError("dataset config names mapping is empty")
    if isinstance(names, list) and not names:
        raise ValueError("dataset config names list is empty")
    if not isinstance(names, (dict, list)):
        raise ValueError("dataset config names must be a dict or list")


def train(config: dict) -> None:
    validate_dataset_config(config["data"])
    model = YOLO(config["model"])
    train_kwargs = {k: v for k, v in config.items() if k in TRAIN_ARG_KEYS}

    print("Training configuration:")
    print(yaml.safe_dump(train_kwargs, sort_keys=False))
    model.train(**train_kwargs)

    trainer = model.trainer
    best_path = None
    if trainer is not None and getattr(trainer, "best", None):
        best_path = Path(str(trainer.best))

    best_out = Path(config["best_weights_out"])
    copy_artifact(best_path, best_out)
    if best_path is not None and best_path.exists():
        print(f"Saved best weights to: {best_out}")
    else:
        print("Warning: best.pt was not found after training.")

    if config.get("export_onnx", False):
        weights_for_export = str(best_path if best_path is not None and best_path.exists() else config["model"])
        onnx_out = Path(config["onnx_out"])
        try:
            export_model = YOLO(weights_for_export)
            exported_path = export_model.export(
                format="onnx",
                imgsz=config.get("imgsz", 960),
                dynamic=True,
                simplify=True,
            )
            exported_path_obj = Path(str(exported_path))
            if exported_path_obj.exists():
                copy_artifact(exported_path_obj, onnx_out)
                print(f"Saved ONNX model to: {onnx_out}")
        except Exception as exc:
            print(f"Warning: ONNX export failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model for defect detection.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
