"""FastAPI inference service for the SwiftCFD surrogate.

Endpoints
---------
POST /predict     validated geometry + flow → Ux, Uy, p fields (422 outside envelope)
GET  /health      liveness / readiness
GET  /model-info  model version, training data summary, input bounds
GET  /metrics     request count, p50/p95 latency, rejection count

Run locally::

    uvicorn service.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .envelope import DEFAULT_ENVELOPE, check_flow, check_mask, check_shape
from .logging_utils import configure_logging, input_hash, log_request
from .metrics import Metrics
from .model import FILTERS, HF_REPO, KERNEL_SIZE, LoadedModel, load_model, predict
from .preprocess import NX, NY, build_input_tensor, shape_to_mask
from .schemas import (FieldStats, HealthResponse, MaskObstacle, MetricsResponse,
                      ModelInfoResponse, PredictRequest, PredictResponse, ShapeObstacle)

OUTPUT_FIELDS = ["Ux", "Uy", "p"]

TRAINING_DATA_SUMMARY = {
    "dataset": "DeepCFD (Ribeiro et al., 2020)",
    "source": "https://zenodo.org/record/3666056",
    "n_samples": 981,
    "split": {"train": 686, "test": 295, "seed": 42},
    "grid": {"nx": NX, "ny": NY},
    "flow_regime": "steady-state laminar, 2D, single obstacle",
    "input_channels": ["SDF", "Ux_inlet", "Uy_inlet"],
    "output_channels": OUTPUT_FIELDS,
    "best_val_mse": 0.739,
    "test_r2": {"Ux": 0.9974, "Uy": 0.9960, "p": 0.7552},
}

logger = configure_logging()
access_log = logging.getLogger("swiftcfd.requests")


class State:
    loaded: LoadedModel | None = None
    metrics = Metrics()
    started_at = time.time()
    load_error: str | None = None


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.environ.get("SWIFTCFD_TORCH_THREADS", "1")))
    try:
        state.loaded = load_model()
        # warm-up: the first forward pass pays one-off allocation costs (~10x slower)
        predict(state.loaded.model, build_input_tensor(shape_to_mask("circle", 55, 39, 10), 0.1))
    except Exception as e:  # keep the process up so /health can report the problem
        state.load_error = f"{type(e).__name__}: {e}"
        logger.error("model load failed", extra={"data": {"event": "model_load_failed",
                                                          "error": state.load_error}})
    yield


app = FastAPI(
    title="SwiftCFD inference service",
    version=__version__,
    description="Neural surrogate for steady laminar flow around a 2D obstacle, "
                "with a training-envelope guard.",
    lifespan=lifespan,
)


# ── Error handling ─────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    """Count schema failures on /predict as rejections and log them."""
    if request.url.path == "/predict":
        state.metrics.record(0.0, "rejected")
        try:
            body = json.loads(await request.body())
            h = input_hash(body) if isinstance(body, dict) else "invalid"
        except Exception:
            h = "invalid"
        log_request(access_log, input_hash=h, latency_ms=0.0, in_envelope=False,
                    status=422, outcome="rejected", reason="schema")
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors()), "reason": "schema"})


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    ok = state.loaded is not None
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=ok,
        model_version=state.loaded.version if ok else None,
        uptime_s=round(time.time() - state.started_at, 3),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if state.loaded is None:
        raise HTTPException(503, detail=f"model not loaded: {state.load_error}")
    return ModelInfoResponse(
        model_version=state.loaded.version,
        weights_source=state.loaded.source if state.loaded.source != "hf" else HF_REPO,
        architecture={
            "name": "UNetEx",
            "filters": FILTERS,
            "kernel_size": KERNEL_SIZE,
            "batch_norm": False,
            "weight_norm": False,
            "n_parameters": state.loaded.n_parameters,
            "input_shape": [3, NX, NY],
            "output_shape": [3, NX, NY],
        },
        training_data=TRAINING_DATA_SUMMARY,
        input_bounds=DEFAULT_ENVELOPE.as_dict(),
        output_fields=OUTPUT_FIELDS,
        service_version=__version__,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return MetricsResponse(**state.metrics.snapshot())


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True,
          responses={422: {"description": "schema error or input outside the training envelope"},
                     503: {"description": "model not loaded"}})
def predict_endpoint(req: PredictRequest):
    t0 = time.perf_counter()
    body = req.model_dump()
    h = input_hash(body)

    if state.loaded is None:
        state.metrics.record(0.0, "error")
        log_request(access_log, input_hash=h, latency_ms=0.0, in_envelope=False,
                    status=503, outcome="error", reason="model_not_loaded")
        raise HTTPException(503, detail=f"model not loaded: {state.load_error}")

    # ── envelope guard ─────────────────────────────────────────────
    violations = check_flow(req.ux_inlet, req.uy_inlet)
    if isinstance(req.obstacle, ShapeObstacle):
        violations += check_shape(req.obstacle.cx, req.obstacle.cy, req.obstacle.size)
        mask = shape_to_mask(req.obstacle.type, req.obstacle.cx, req.obstacle.cy, req.obstacle.size)
    else:
        assert isinstance(req.obstacle, MaskObstacle)
        mask = np.asarray(req.obstacle.mask, dtype=np.uint8)
    violations += check_mask(mask)

    if violations:
        latency = (time.perf_counter() - t0) * 1000
        state.metrics.record(latency, "rejected")
        log_request(access_log, input_hash=h, latency_ms=round(latency, 3), in_envelope=False,
                    status=422, outcome="rejected", reason="envelope",
                    violations=[v["field"] for v in violations])
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "outside_training_envelope",
                "message": ("Input is outside the envelope the model was trained on, so the "
                            "prediction would be a silent extrapolation. "
                            + "; ".join(v["message"] for v in violations)
                            + ". See GET /model-info for the allowed bounds."),
                "violations": violations,
            },
        )

    # ── inference ──────────────────────────────────────────────────
    try:
        x = build_input_tensor(mask, req.ux_inlet, req.uy_inlet)
        t_inf = time.perf_counter()
        y = predict(state.loaded.model, x).squeeze(0).numpy()
        inference_ms = (time.perf_counter() - t_inf) * 1000
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        state.metrics.record(latency, "error")
        log_request(access_log, input_hash=h, latency_ms=round(latency, 3), in_envelope=True,
                    status=500, outcome="error", reason=type(e).__name__)
        raise HTTPException(500, detail=f"inference failed: {type(e).__name__}") from e

    stats = {
        name: FieldStats(min=float(y[i].min()), max=float(y[i].max()),
                         mean=float(y[i].mean()), std=float(y[i].std()))
        for i, name in enumerate(OUTPUT_FIELDS)
    }
    fields = ({name: y[i].astype(float).round(6).tolist() for i, name in enumerate(OUTPUT_FIELDS)}
              if req.include_fields else None)

    latency = (time.perf_counter() - t0) * 1000
    state.metrics.record(latency, "ok")
    log_request(access_log, input_hash=h, latency_ms=round(latency, 3),
                inference_ms=round(inference_ms, 3), in_envelope=True, status=200, outcome="ok")

    return PredictResponse(
        model_version=state.loaded.version,
        input_hash=h,
        in_envelope=True,
        inference_time_ms=round(inference_ms, 3),
        grid={"nx": NX, "ny": NY},
        stats=stats,
        fields=fields,
    )
