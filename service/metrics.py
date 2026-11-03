"""In-process request metrics (thread-safe, bounded memory)."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class Metrics:
    def __init__(self, window: int = 10_000):
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self.window = window
        self.request_count = 0     # every /predict call
        self.prediction_count = 0  # successful predictions
        self.rejection_count = 0   # 422s: schema or envelope
        self.error_count = 0       # 5xx

    def record(self, latency_ms: float, outcome: str) -> None:
        with self._lock:
            self.request_count += 1
            if outcome == "ok":
                self.prediction_count += 1
                self._latencies.append(latency_ms)
            elif outcome == "rejected":
                self.rejection_count += 1
            else:
                self.error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = np.fromiter(self._latencies, dtype=float)
            p50 = float(np.percentile(lat, 50)) if lat.size else None
            p95 = float(np.percentile(lat, 95)) if lat.size else None
            return {
                "request_count": self.request_count,
                "prediction_count": self.prediction_count,
                "rejection_count": self.rejection_count,
                "error_count": self.error_count,
                "latency_ms_p50": p50,
                "latency_ms_p95": p95,
                "window": self.window,
            }

    def reset(self) -> None:
        with self._lock:
            self._latencies.clear()
            self.request_count = self.prediction_count = 0
            self.rejection_count = self.error_count = 0
