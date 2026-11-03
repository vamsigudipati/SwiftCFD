"""Shared fixtures.

The tests run against deterministic *fixture* weights (seeded random init of
the real UNetEx architecture) so they need no network access and no model
download. Set ``SWIFTCFD_WEIGHTS=/path/to/mymodel_v2.pt`` to run the same
suite against the real model; the stored-reference smoke test is then skipped
unless the reference file was regenerated for that model
(``python tests/make_reference.py``).
"""

import os

os.environ.setdefault("SWIFTCFD_WEIGHTS", "fixture")
os.environ.setdefault("SWIFTCFD_LOG_LEVEL", "WARNING")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from service.main import app, state  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_metrics():
    state.metrics.reset()
    yield


@pytest.fixture
def good_request():
    return {"obstacle": {"type": "circle", "cx": 55, "cy": 39, "size": 10}, "ux_inlet": 0.1}
