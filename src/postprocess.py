from __future__ import annotations

from uuid import uuid4

import cv2
import numpy as np
from ultralytics.engine.results import Results

from src.calibration import CalibrationResult, area_mm2, severity_bucket


def detections_from_result(
    result: Results,
    class_names: dict[int, str],
    calibration: CalibrationResult | None = None,
    conf_threshold: float = 0.0,
    include_severity: bool = False,
) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    image_h, image_w = result.orig_shape

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    boxes_conf = result.boxes.conf.detach().cpu().numpy()
    boxes_cls = result.boxes.cls.detach().cpu().numpy().astype(int)

    masks_data = None
    if result.masks is not None and result.masks.data is not None:
        masks_data = result.masks.data.detach().cpu().numpy()

    detections: list[dict] = []
    for idx, (xyxy, conf, cls_idx) in enumerate(zip(boxes_xyxy, boxes_conf, boxes_cls)):
        score = float(conf)
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in xyxy.tolist()]
        bbox = [max(0, x1), max(0, y1), min(image_w, x2), min(image_h, y2)]

        if masks_data is not None and idx < len(masks_data):
            mask = masks_data[idx]
            if mask.shape[:2] != (image_h, image_w):
                mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
            mask_area_px = int(np.count_nonzero(mask > 0.5))
        else:
            box_w = max(0, bbox[2] - bbox[0])
            box_h = max(0, bbox[3] - bbox[1])
            mask_area_px = box_w * box_h

        calibration_payload = None
        area_mm2_value = None
        if calibration is not None:
            area_mm2_value = round(area_mm2(mask_area_px, calibration.mm_per_px), 4)
            calibration_payload = {
                "method": calibration.method,
                "mm_per_px": round(calibration.mm_per_px, 8),
            }

        detection = {
            "id": str(uuid4()),
            "class": class_names.get(int(cls_idx), str(cls_idx)),
            "confidence": round(score, 4),
            "bbox": bbox,
            "mask_area_px": mask_area_px,
            "area_mm2": area_mm2_value,
            "calibration": calibration_payload,
        }

        if include_severity:
            detection["severity"] = severity_bucket(mask_area_px)

        detections.append(detection)

    return detections

