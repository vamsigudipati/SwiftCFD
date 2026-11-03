"""Pydantic schema validation on POST /predict."""

import pytest

from service.preprocess import NX, NY


def test_valid_shape_request_accepted(client, good_request):
    r = client.post("/predict", json={**good_request, "include_fields": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["in_envelope"] is True
    assert set(body["stats"]) == {"Ux", "Uy", "p"}
    assert "fields" not in body  # include_fields=False and exclude_none


def test_full_fields_have_grid_shape(client, good_request):
    r = client.post("/predict", json=good_request)
    assert r.status_code == 200
    fields = r.json()["fields"]
    for name in ("Ux", "Uy", "p"):
        assert len(fields[name]) == NX
        assert all(len(row) == NY for row in fields[name])


@pytest.mark.parametrize("bad", [
    {},                                                        # nothing
    {"ux_inlet": 0.1},                                         # missing obstacle
    {"obstacle": {"type": "circle", "cx": 55, "cy": 39, "size": 10}},   # missing ux_inlet
    {"obstacle": {"type": "hexagon", "cx": 55, "cy": 39, "size": 10}, "ux_inlet": 0.1},
    {"obstacle": {"type": "circle", "cx": -1, "cy": 39, "size": 10}, "ux_inlet": 0.1},
    {"obstacle": {"type": "circle", "cx": 55, "cy": NY, "size": 10}, "ux_inlet": 0.1},
    {"obstacle": {"type": "circle", "cx": 55, "cy": 39, "size": 0}, "ux_inlet": 0.1},
    {"obstacle": {"type": "circle", "cx": "a", "cy": 39, "size": 10}, "ux_inlet": 0.1},
    {"obstacle": {"type": "circle", "cx": 55, "cy": 39, "size": 10}, "ux_inlet": "fast"},
])
def test_schema_errors_are_422(client, bad):
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert r.json()["reason"] == "schema"


def test_mask_wrong_shape_rejected(client):
    r = client.post("/predict", json={"obstacle": {"type": "mask", "mask": [[1, 0], [0, 1]]},
                                      "ux_inlet": 0.1})
    assert r.status_code == 422
    assert "shape" in r.text


def test_mask_non_binary_rejected(client):
    mask = [[0] * NY for _ in range(NX)]
    mask[50][40] = 2
    r = client.post("/predict", json={"obstacle": {"type": "mask", "mask": mask}, "ux_inlet": 0.1})
    assert r.status_code == 422
    assert "0 or 1" in r.text


def test_schema_rejection_is_counted(client):
    client.post("/predict", json={})
    m = client.get("/metrics").json()
    assert m["rejection_count"] == 1
    assert m["prediction_count"] == 0
