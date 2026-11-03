"""Prediction smoke test: fixed input vs stored reference output."""

import json
from pathlib import Path

import numpy as np
import pytest

from service.main import state

REFERENCE = Path(__file__).parent / "fixtures" / "reference_prediction.json"
RTOL, ATOL = 1e-3, 1e-5


@pytest.fixture(scope="module")
def reference():
    return json.loads(REFERENCE.read_text())


def test_prediction_matches_reference(client, reference):
    if state.loaded.version != reference["model_version"]:
        pytest.skip(f"reference was generated for {reference['model_version']}, "
                    f"service runs {state.loaded.version}; run tests/make_reference.py")

    r = client.post("/predict", json=reference["input"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] == reference["model_version"]
    assert body["inference_time_ms"] > 0

    for name, ref in reference["fields"].items():
        got = np.asarray(body["fields"][name], dtype=float)
        stats = body["stats"][name]
        for k in ("min", "max", "mean", "std"):
            assert stats[k] == pytest.approx(ref[k], rel=RTOL, abs=ATOL), f"{name}.{k}"
        sampled = got[np.ix_(reference["sample_x"], reference["sample_y"])]
        np.testing.assert_allclose(sampled, np.asarray(ref["samples"]), rtol=RTOL, atol=ATOL,
                                   err_msg=f"{name} sampled values drifted from reference")


def test_prediction_is_deterministic(client, reference):
    a = client.post("/predict", json=reference["input"]).json()
    b = client.post("/predict", json=reference["input"]).json()
    assert a["input_hash"] == b["input_hash"]
    assert a["fields"] == b["fields"]


def test_outputs_are_finite(client, reference):
    body = client.post("/predict", json=reference["input"]).json()
    for name in ("Ux", "Uy", "p"):
        assert np.isfinite(np.asarray(body["fields"][name])).all()
