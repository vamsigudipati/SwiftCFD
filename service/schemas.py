"""Pydantic request / response models for the inference API."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

from .preprocess import NX, NY


class ShapeObstacle(BaseModel):
    """A geometric primitive, rasterised server-side (same rules as the app)."""

    type: Literal["rectangle", "circle", "diamond", "triangle"]
    cx: int = Field(..., ge=0, le=NX - 1,
                    description=f"centre along the streamwise axis, 0..{NX - 1}")
    cy: int = Field(..., ge=0, le=NY - 1,
                    description=f"centre along the transverse axis, 0..{NY - 1}")
    size: int = Field(..., ge=1, le=min(NX, NY) // 2,
                      description="radius / half-width in cells")


class MaskObstacle(BaseModel):
    """An explicit binary mask, e.g. from a drawing tool."""

    type: Literal["mask"]
    mask: list[list[int]] = Field(
        ..., description=f"binary mask of shape ({NX}, {NY}); mask[x][y] == 1 inside the obstacle")

    @field_validator("mask")
    @classmethod
    def _check_shape_and_values(cls, m: list[list[int]]) -> list[list[int]]:
        if len(m) != NX or any(len(row) != NY for row in m):
            raise ValueError(f"mask must have shape ({NX}, {NY})")
        for row in m:
            for v in row:
                if v not in (0, 1):
                    raise ValueError("mask values must be 0 or 1")
        return m


Obstacle = Annotated[Union[ShapeObstacle, MaskObstacle], Field(discriminator="type")]


class PredictRequest(BaseModel):
    obstacle: Obstacle
    ux_inlet: float = Field(..., description="inlet streamwise velocity, m/s")
    uy_inlet: float = Field(0.0, description="inlet transverse velocity, m/s (0 in training data)")
    include_fields: bool = Field(
        True, description="return the full (172×79) Ux/Uy/p fields; False returns statistics only")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "obstacle": {"type": "circle", "cx": 55, "cy": 39, "size": 10},
                "ux_inlet": 0.1,
            }]
        }
    }


class FieldStats(BaseModel):
    min: float
    max: float
    mean: float
    std: float


class PredictResponse(BaseModel):
    model_version: str
    input_hash: str
    in_envelope: bool
    inference_time_ms: float
    grid: dict[str, int]
    stats: dict[str, FieldStats]
    fields: dict[str, list[list[float]]] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    uptime_s: float


class ModelInfoResponse(BaseModel):
    model_version: str
    weights_source: str
    architecture: dict
    training_data: dict
    input_bounds: dict
    output_fields: list[str]
    service_version: str


class MetricsResponse(BaseModel):
    request_count: int
    prediction_count: int
    rejection_count: int
    error_count: int
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    window: int
