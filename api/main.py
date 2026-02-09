from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.infer import DefectInferencer, is_ultralytics_model_alias
from src.schemas import InferenceResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inferencer = DefectInferencer(
        weights_path=os.getenv("MODEL_WEIGHTS", "models/weights/best.pt"),
        conf=float(os.getenv("CONF_THRESH", "0.25")),
        iou=float(os.getenv("IOU_THRESH", "0.5")),
        device=os.getenv("DEVICE", "cpu"),
    )
    yield


app = FastAPI(title="Defect Detector API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    inferencer = app.state.inferencer
    local_path = Path(inferencer.weights_path)
    alias = is_ultralytics_model_alias(inferencer.weights_path)
    local_exists = local_path.exists()

    status = "ok"
    if not alias and not local_exists:
        status = "degraded"

    return {
        "status": status,
        "model": inferencer.weights_path,
        "model_loaded": inferencer.is_model_loaded,
        "weights_type": "alias" if alias else "local",
        "weights_found": local_exists if not alias else None,
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    fixed_mm_per_px: float | None = Form(default=None),
    marker_size_mm: float | None = Form(default=None),
    include_severity: bool = Form(default=False),
) -> dict:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        response = app.state.inferencer.infer_bytes(
            image_bytes=image_bytes,
            fixed_mm_per_px=fixed_mm_per_px,
            marker_size_mm=marker_size_mm,
            include_severity=include_severity,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    return response.model_dump(by_alias=True)
