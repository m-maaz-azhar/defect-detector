from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from ultralytics import YOLO

from src.calibration import fixed_scale, marker_scale_from_image
from src.postprocess import detections_from_result
from src.schemas import InferenceResponse


ULTRALYTICS_MODEL_PREFIXES: Final[tuple[str, ...]] = ("yolo", "sam", "fastsam", "rtdetr")


def is_ultralytics_model_alias(weights_path: str | Path) -> bool:
    raw = str(weights_path).strip()
    if not raw:
        return False
    if "/" in raw or "\\" in raw:
        return False
    return raw.lower().startswith(ULTRALYTICS_MODEL_PREFIXES)


class DefectInferencer:
    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.weights_path = str(weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model: YOLO | None = None
        self._validate_threshold("conf", self.conf)
        self._validate_threshold("iou", self.iou)

    @staticmethod
    def _validate_threshold(name: str, value: float) -> None:
        if not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be in (0, 1], got {value}")

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    def _resolve_weights_for_loading(self) -> str:
        path_obj = Path(self.weights_path)
        if path_obj.exists():
            return str(path_obj)
        if is_ultralytics_model_alias(self.weights_path):
            return self.weights_path
        raise FileNotFoundError(
            f"weights not found at '{self.weights_path}'. "
            "Set MODEL_WEIGHTS to an existing file or a valid Ultralytics alias "
            "(example: yolov8m-seg.pt)."
        )

    def _get_model(self) -> YOLO:
        if self._model is None:
            resolved = self._resolve_weights_for_loading()
            self._model = YOLO(resolved)
        return self._model

    def infer_file(
        self,
        image_path: str | Path,
        fixed_mm_per_px: float | None = None,
        marker_size_mm: float | None = None,
        include_severity: bool = False,
    ) -> InferenceResponse:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        return self.infer_array(
            image_bgr=image,
            fixed_mm_per_px=fixed_mm_per_px,
            marker_size_mm=marker_size_mm,
            include_severity=include_severity,
        )

    def infer_bytes(
        self,
        image_bytes: bytes,
        fixed_mm_per_px: float | None = None,
        marker_size_mm: float | None = None,
        include_severity: bool = False,
    ) -> InferenceResponse:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("uploaded bytes are not a valid image")
        return self.infer_array(
            image_bgr=image,
            fixed_mm_per_px=fixed_mm_per_px,
            marker_size_mm=marker_size_mm,
            include_severity=include_severity,
        )

    def infer_array(
        self,
        image_bgr: np.ndarray,
        fixed_mm_per_px: float | None = None,
        marker_size_mm: float | None = None,
        include_severity: bool = False,
    ) -> InferenceResponse:
        if fixed_mm_per_px is not None and marker_size_mm is not None:
            raise ValueError("provide either fixed_mm_per_px or marker_size_mm, not both")
        self._validate_threshold("conf", self.conf)
        self._validate_threshold("iou", self.iou)

        calibration = None
        if fixed_mm_per_px is not None:
            calibration = fixed_scale(fixed_mm_per_px)
        elif marker_size_mm is not None:
            calibration = marker_scale_from_image(image_bgr, marker_size_mm)

        prediction = self._get_model().predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )[0]

        class_names = self._class_names()
        defects = detections_from_result(
            result=prediction,
            class_names=class_names,
            calibration=calibration,
            conf_threshold=self.conf,
            include_severity=include_severity,
        )

        payload = {
            "model": Path(self.weights_path).name,
            "image_width": int(image_bgr.shape[1]),
            "image_height": int(image_bgr.shape[0]),
            "defects": defects,
        }
        return InferenceResponse.model_validate(payload)

    def _class_names(self) -> dict[int, str]:
        names = self._get_model().names
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {idx: str(name) for idx, name in enumerate(names)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defect segmentation inference on one image.")
    parser.add_argument("--weights", type=str, default="models/weights/best.pt", help="Path to YOLO weights.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument("--device", type=str, default="cpu", help='Device (e.g. "cpu", "0").')
    parser.add_argument("--fixed-mm-per-px", type=float, default=None, help="Use fixed calibration scale.")
    parser.add_argument("--marker-size-mm", type=float, default=None, help="Known marker size in mm.")
    parser.add_argument(
        "--include-severity",
        action="store_true",
        help="Add small/medium/large severity bucket based on pixel area.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inferencer = DefectInferencer(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    response = inferencer.infer_file(
        image_path=args.image,
        fixed_mm_per_px=args.fixed_mm_per_px,
        marker_size_mm=args.marker_size_mm,
        include_severity=args.include_severity,
    )
    print(
        json.dumps(
            response.model_dump(by_alias=True),
            indent=2 if args.pretty else None,
        )
    )


if __name__ == "__main__":
    main()
