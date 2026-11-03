"""Regenerate tests/fixtures/reference_prediction.json.

Runs the fixed smoke-test input through whichever weights
``SWIFTCFD_WEIGHTS`` selects (default: fixture) and stores per-field
statistics plus a coarse grid of sampled values. The smoke test compares a
live prediction against this file within a tolerance.

    SWIFTCFD_WEIGHTS=fixture python tests/make_reference.py
    SWIFTCFD_WEIGHTS=/path/to/mymodel_v2.pt python tests/make_reference.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from service.model import load_model, predict  # noqa: E402
from service.preprocess import build_input_tensor, shape_to_mask  # noqa: E402

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "reference_prediction.json"
SMOKE_INPUT = {"obstacle": {"type": "circle", "cx": 55, "cy": 39, "size": 10}, "ux_inlet": 0.1}
SAMPLE_X = [10, 40, 55, 70, 100, 160]
SAMPLE_Y = [5, 20, 39, 60, 75]
FIELDS = ["Ux", "Uy", "p"]


def main() -> None:
    loaded = load_model(os.environ.get("SWIFTCFD_WEIGHTS", "fixture"))
    o = SMOKE_INPUT["obstacle"]
    x = build_input_tensor(shape_to_mask(o["type"], o["cx"], o["cy"], o["size"]), SMOKE_INPUT["ux_inlet"])
    y = predict(loaded.model, x).squeeze(0).numpy().astype(np.float64)

    ref = {
        "model_version": loaded.version,
        "input": SMOKE_INPUT,
        "sample_x": SAMPLE_X,
        "sample_y": SAMPLE_Y,
        "fields": {},
    }
    for i, name in enumerate(FIELDS):
        f = y[i]
        ref["fields"][name] = {
            "min": float(f.min()), "max": float(f.max()),
            "mean": float(f.mean()), "std": float(f.std()),
            "samples": [[round(float(f[ix, iy]), 7) for iy in SAMPLE_Y] for ix in SAMPLE_X],
        }
    REFERENCE_PATH.write_text(json.dumps(ref, indent=1) + "\n")
    print(f"wrote {REFERENCE_PATH} for model {loaded.version}")


if __name__ == "__main__":
    main()
