"""Geometry → model-input preprocessing.

Mirrors ``compute_sdf`` / ``build_input_tensor`` / ``shape_to_mask`` in
``app.py`` so the service produces exactly the same tensors the demo does.

Axis convention
---------------
The model works on arrays of shape ``(3, NX, NY) = (3, 172, 79)``.
Index 1 (length 172) is the **streamwise x** direction; index 2 (length 79)
is the **transverse y** direction. A mask is therefore ``mask[x, y]``.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import torch

NX, NY = 172, 79  # grid size the model was trained on


def compute_sdf(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(float)
    d_out = ndimage.distance_transform_edt(1 - mask)
    d_in = ndimage.distance_transform_edt(mask)
    sdf = d_out - d_in
    maxval = max(abs(sdf).max(), 1e-8)
    return (sdf / maxval).astype(np.float32)


def build_input_tensor(mask: np.ndarray, ux_inlet: float, uy_inlet: float = 0.0) -> torch.Tensor:
    sdf = compute_sdf(mask)
    ux = np.ones(mask.shape, dtype=np.float32) * ux_inlet
    uy = np.ones(mask.shape, dtype=np.float32) * uy_inlet
    ux[mask == 1] = 0.0
    uy[mask == 1] = 0.0
    return torch.FloatTensor(np.stack([sdf, ux, uy], axis=0))


def shape_to_mask(shape_type: str, cx: int, cy: int, size: int,
                  nx: int = NX, ny: int = NY) -> np.ndarray:
    """Rasterise a primitive centred at ``(cx, cy)`` into a ``(nx, ny)`` mask.

    Same geometry rules as ``app.py``: ``size`` is the radius for circle and
    diamond, the half-width for triangle, and (as in the app) a rectangle is
    ``2*size`` long along the first axis and ``size`` wide along the second.
    """
    mask = np.zeros((nx, ny), dtype=np.uint8)
    X, Y = np.ogrid[:nx, :ny]
    if shape_type == "rectangle":
        mask[max(0, cx - size):min(nx, cx + size),
             max(0, cy - size // 2):min(ny, cy + size // 2)] = 1
    elif shape_type == "circle":
        mask[np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) <= size] = 1
    elif shape_type == "diamond":
        mask[np.abs(X - cx) + np.abs(Y - cy) <= size] = 1
    elif shape_type == "triangle":
        # isosceles triangle pointing in +x, apex at cx + size
        inside = (X >= cx - size) & (X <= cx + size) & (np.abs(Y - cy) <= size - np.abs(X - cx))
        mask[inside] = 1
    else:  # pragma: no cover - guarded by the schema
        raise ValueError(f"unknown shape {shape_type!r}")
    return mask


def mask_summary(mask: np.ndarray) -> dict:
    """Centroid, bounding box, area fraction and component count of a mask."""
    xs, ys = np.nonzero(mask)
    if xs.size == 0:
        return {"empty": True, "area_fraction": 0.0, "n_components": 0}
    _, n_components = ndimage.label(mask)
    return {
        "empty": False,
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "bbox": {"x_min": int(xs.min()), "x_max": int(xs.max()),
                 "y_min": int(ys.min()), "y_max": int(ys.max())},
        "area_fraction": float(mask.sum() / mask.size),
        "n_components": int(n_components),
    }
