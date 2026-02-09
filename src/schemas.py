from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CalibrationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["marker", "fixed"]
    mm_per_px: float = Field(gt=0)


class DefectPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str
    class_: str = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    bbox: list[int] = Field(min_length=4, max_length=4)
    mask_area_px: int = Field(ge=0)
    area_mm2: float | None = Field(default=None, ge=0)
    calibration: CalibrationInfo | None = None
    severity: Literal["small", "medium", "large"] | None = None


class InferenceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    model: str
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    defects: list[DefectPrediction] = Field(default_factory=list)

