"""Training-envelope guard.

The surrogate was trained on 981 DeepCFD samples: a single obstacle placed in
the left-centre third of a 172 × 79 domain with a modest inlet velocity. The
network has no notion of "I have not seen this before" — for an obstacle
outside that region it quietly returns an averaged, physically meaningless
field (see DOCUMENTATION.md, Section 8, Issue 5). The service therefore
refuses such inputs with HTTP 422 rather than returning a number that looks
plausible but is not.

The bounds below are taken from README.md "Known Limitations" and
DOCUMENTATION.md. They are deliberately a single, explicit table so they can
be re-derived from ``dataX.pkl`` and tightened or relaxed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .preprocess import NX, NY, mask_summary


@dataclass(frozen=True)
class Bounds:
    lo: float
    hi: float

    def contains(self, v: float) -> bool:
        return self.lo <= v <= self.hi

    def as_dict(self) -> dict:
        return {"min": self.lo, "max": self.hi}


@dataclass(frozen=True)
class Envelope:
    # obstacle centre along the streamwise axis (index 1, length 172)
    obstacle_center_x: Bounds = field(default_factory=lambda: Bounds(30, 80))
    # obstacle centre along the transverse axis (index 2, length 79)
    obstacle_center_y: Bounds = field(default_factory=lambda: Bounds(20, 58))
    # radius / half-width of the primitive, in cells
    obstacle_size: Bounds = field(default_factory=lambda: Bounds(3, 20))
    # fraction of the domain covered by the obstacle (for mask inputs)
    obstacle_area_fraction: Bounds = field(default_factory=lambda: Bounds(0.002, 0.15))
    # inlet velocities, m/s. Uy is effectively zero in the training data.
    ux_inlet: Bounds = field(default_factory=lambda: Bounds(0.01, 0.30))
    uy_inlet: Bounds = field(default_factory=lambda: Bounds(0.0, 0.0))
    max_components: int = 1  # single obstacle only
    grid: tuple[int, int] = (NX, NY)

    def as_dict(self) -> dict:
        return {
            "grid": {"nx": self.grid[0], "ny": self.grid[1]},
            "obstacle_center_x": self.obstacle_center_x.as_dict(),
            "obstacle_center_y": self.obstacle_center_y.as_dict(),
            "obstacle_size": self.obstacle_size.as_dict(),
            "obstacle_area_fraction": self.obstacle_area_fraction.as_dict(),
            "ux_inlet": self.ux_inlet.as_dict(),
            "uy_inlet": self.uy_inlet.as_dict(),
            "max_obstacles": self.max_components,
        }


DEFAULT_ENVELOPE = Envelope()


class EnvelopeViolation(ValueError):
    """Raised when a request lies outside the training envelope."""

    def __init__(self, violations: list[dict]):
        self.violations = violations
        msg = "; ".join(v["message"] for v in violations)
        super().__init__(msg)


def _check(violations: list, field_name: str, value: float, bounds: Bounds, what: str) -> None:
    if not bounds.contains(value):
        violations.append({
            "field": field_name,
            "value": value,
            "allowed": bounds.as_dict(),
            "message": (f"{what} {value:g} is outside the training envelope "
                        f"[{bounds.lo:g}, {bounds.hi:g}]"),
        })


def check_flow(ux_inlet: float, uy_inlet: float, env: Envelope = DEFAULT_ENVELOPE) -> list[dict]:
    v: list[dict] = []
    _check(v, "ux_inlet", ux_inlet, env.ux_inlet, "inlet velocity ux_inlet")
    _check(v, "uy_inlet", uy_inlet, env.uy_inlet, "inlet velocity uy_inlet")
    return v


def check_shape(cx: int, cy: int, size: int, env: Envelope = DEFAULT_ENVELOPE) -> list[dict]:
    v: list[dict] = []
    _check(v, "obstacle.cx", cx, env.obstacle_center_x, "obstacle centre x")
    _check(v, "obstacle.cy", cy, env.obstacle_center_y, "obstacle centre y")
    _check(v, "obstacle.size", size, env.obstacle_size, "obstacle size")
    return v


def check_mask(mask, env: Envelope = DEFAULT_ENVELOPE) -> list[dict]:
    v: list[dict] = []
    s = mask_summary(mask)
    if s["empty"]:
        v.append({"field": "obstacle.mask", "value": 0, "allowed": None,
                  "message": "obstacle mask is empty; the model needs exactly one obstacle"})
        return v
    if s["n_components"] > env.max_components:
        v.append({"field": "obstacle.mask", "value": s["n_components"],
                  "allowed": {"max": env.max_components},
                  "message": (f"mask contains {s['n_components']} separate obstacles; "
                              f"the model was trained on a single obstacle")})
    _check(v, "obstacle.mask.centroid_x", round(s["centroid_x"], 2),
           env.obstacle_center_x, "obstacle centroid x")
    _check(v, "obstacle.mask.centroid_y", round(s["centroid_y"], 2),
           env.obstacle_center_y, "obstacle centroid y")
    _check(v, "obstacle.mask.area_fraction", round(s["area_fraction"], 4),
           env.obstacle_area_fraction, "obstacle area fraction")
    return v
