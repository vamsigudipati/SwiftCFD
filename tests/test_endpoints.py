"""Health, model-info and metrics endpoints; JSON-lines request logging."""

import json
import logging

from service import __version__
from service.logging_utils import input_hash


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"]
    assert body["uptime_s"] >= 0


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["architecture"]["name"] == "UNetEx"
    assert body["architecture"]["filters"] == [8, 16, 32, 32]
    assert body["architecture"]["input_shape"] == [3, 172, 79]
    assert body["training_data"]["n_samples"] == 981
    assert body["output_fields"] == ["Ux", "Uy", "p"]
    assert body["service_version"] == __version__
    assert "ux_inlet" in body["input_bounds"]


def test_metrics_empty(client):
    m = client.get("/metrics").json()
    assert m["request_count"] == 0
    assert m["latency_ms_p50"] is None and m["latency_ms_p95"] is None


def test_metrics_after_traffic(client, good_request):
    for _ in range(3):
        assert client.post("/predict", json={**good_request, "include_fields": False}).status_code == 200
    assert client.post("/predict", json={**good_request, "ux_inlet": 5.0}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    m = client.get("/metrics").json()
    assert m["request_count"] == 5
    assert m["prediction_count"] == 3
    assert m["rejection_count"] == 2
    assert m["error_count"] == 0
    assert 0 < m["latency_ms_p50"] <= m["latency_ms_p95"]


def test_request_log_is_json_lines(client, good_request, caplog):
    with caplog.at_level(logging.INFO, logger="swiftcfd.requests"):
        client.post("/predict", json={**good_request, "include_fields": False})
        client.post("/predict", json={**good_request, "ux_inlet": 5.0})
    records = [r.data for r in caplog.records if getattr(r, "data", None)]
    assert len(records) == 2
    ok, rejected = records
    for rec in records:
        json.dumps(rec)  # serialisable
        assert {"event", "input_hash", "latency_ms", "in_envelope", "status"} <= rec.keys()
    assert ok["in_envelope"] is True and ok["status"] == 200
    assert rejected["in_envelope"] is False and rejected["status"] == 422
    assert rejected["reason"] == "envelope"


def test_input_hash_is_stable_and_order_independent():
    a = input_hash({"b": 1, "a": {"y": 2, "x": 1}})
    b = input_hash({"a": {"x": 1, "y": 2}, "b": 1})
    assert a == b and len(a) == 16
    assert input_hash({"b": 2}) != a
