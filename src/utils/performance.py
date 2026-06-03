"""Performance monitoring utilities."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class PerformanceMonitor:
    """Track inference latency and throughput."""

    window_size: int = 100
    latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    frame_count: int = 0
    start_time: float = field(default_factory=time.time)

    def record_inference(self, latency_ms: float) -> None:
        self.latencies.append(latency_ms)
        self.frame_count += 1

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def fps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.frame_count / elapsed

    def summary(self) -> Dict[str, float]:
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "fps": round(self.fps, 2),
            "frames_processed": self.frame_count,
        }


class PredictionSmoother:
    """Smooth predictions over a sliding window to reduce flicker."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: Deque[str] = deque(maxlen=window_size)
        self.confidence_history: Deque[float] = deque(maxlen=window_size)

    def add(self, label: str, confidence: float) -> Optional[str]:
        self.history.append(label)
        self.confidence_history.append(confidence)
        if len(self.history) < self.window_size:
            return None
        counts: Dict[str, int] = {}
        for lbl in self.history:
            counts[lbl] = counts.get(lbl, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] >= self.window_size * 0.6:
            return best
        return None

    def reset(self) -> None:
        self.history.clear()
        self.confidence_history.clear()
