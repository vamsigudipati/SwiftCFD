"""Inputs outside the training envelope must be rejected with a clear 422."""

import numpy as np
import pytest

from service.envelope import DEFAULT_ENVELOPE, check_flow, check_mask, check_shape
from service.preprocess import NX, NY, shape_to_mask


def _reject(client, payload, expected_field):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "outside_training_envelope"
    assert "silent extrapolation" in detail["message"]
    assert expected_field in {v["field"] for v in detail["violations"]}
    return detail


def test_obstacle_too_far_downstream(client, good_request):
    payload = {**good_request, "obstacle": {"type": "circle", "cx": 150, "cy": 39, "size": 10}}
    d = _reject(client, payload, "obstacle.cx")
    v = next(v for v in d["violations"] if v["field"] == "obstacle.cx")
    assert v["allowed"] == DEFAULT_ENVELOPE.obstacle_center_x.as_dict()
    assert "150" in v["message"]


def test_obstacle_too_close_to_wall(client, good_request):
    payload = {**good_request, "obstacle": {"type": "circle", "cx": 55, "cy": 5, "size": 4}}
    _reject(client, payload, "obstacle.cy")


def test_obstacle_too_large(client, good_request):
    payload = {**good_request, "obstacle": {"type": "circle", "cx": 55, "cy": 39, "size": 30}}
    _reject(client, payload, "obstacle.size")


@pytest.mark.parametrize("ux", [0.0, 0.5, -0.1])
def test_inlet_velocity_out_of_range(client, good_request, ux):
    _reject(client, {**good_request, "ux_inlet": ux}, "ux_inlet")


def test_transverse_inlet_velocity_rejected(client, good_request):
    _reject(client, {**good_request, "uy_inlet": 0.05}, "uy_inlet")


def test_empty_mask_rejected(client):
    mask = [[0] * NY for _ in range(NX)]
    _reject(client, {"obstacle": {"type": "mask", "mask": mask}, "ux_inlet": 0.1}, "obstacle.mask")


def test_two_obstacles_rejected(client):
    m = shape_to_mask("circle", 45, 39, 6) | shape_to_mask("circle", 70, 39, 6)
    assert m.sum() > 0
    d = _reject(client, {"obstacle": {"type": "mask", "mask": m.tolist()}, "ux_inlet": 0.1},
                "obstacle.mask")
    assert "2 separate obstacles" in d["message"]


def test_in_envelope_mask_accepted(client):
    m = shape_to_mask("diamond", 60, 39, 8)
    r = client.post("/predict", json={"obstacle": {"type": "mask", "mask": m.tolist()},
                                      "ux_inlet": 0.2, "include_fields": False})
    assert r.status_code == 200, r.text
    assert r.json()["in_envelope"] is True


def test_envelope_rejection_is_counted(client, good_request):
    client.post("/predict", json={**good_request, "ux_inlet": 9.0})
    m = client.get("/metrics").json()
    assert m["rejection_count"] == 1 and m["request_count"] == 1


# ── pure-function checks ───────────────────────────────────────────

def test_boundaries_are_inclusive():
    env = DEFAULT_ENVELOPE
    assert check_flow(env.ux_inlet.lo, 0.0) == []
    assert check_flow(env.ux_inlet.hi, 0.0) == []
    assert check_shape(int(env.obstacle_center_x.lo), 39, 10) == []
    assert check_shape(int(env.obstacle_center_x.hi), 39, 10) == []


def test_check_mask_reports_centroid():
    m = shape_to_mask("circle", 55, 39, 10)
    assert check_mask(m) == []
    far = np.roll(m, 90, axis=0)
    fields = {v["field"] for v in check_mask(far)}
    assert "obstacle.mask.centroid_x" in fields


def test_bounds_exposed_in_model_info(client):
    bounds = client.get("/model-info").json()["input_bounds"]
    assert bounds == DEFAULT_ENVELOPE.as_dict()
