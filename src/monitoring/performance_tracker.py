"""Track model accuracy, F1, precision, and recall over sliding time windows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    """Performance metrics at a point in time."""

    timestamp: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    n_samples: int
    window_size: int


@dataclass
class PerformanceDegradation:
    """Record of detected performance degradation."""

    metric: str
    current_value: float
    baseline_value: float
    drop: float
    threshold: float
    is_degraded: bool


class PerformanceTracker:
    """Track model performance metrics over a sliding window of predictions.

    Maintains a rolling buffer of (y_true, y_pred) pairs and computes
    classification metrics. Detects performance degradation by comparing
    current window metrics against a baseline.

    Example:
        >>> tracker = PerformanceTracker(window_size=500)
        >>> tracker.record(y_true=[1, 0, 1], y_pred=[1, 0, 0])
        >>> snapshot = tracker.get_current_metrics()
        >>> print(f"F1: {snapshot.f1:.4f}")
    """

    def __init__(
        self,
        window_size: int | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ) -> None:
        cfg = settings.performance_config
        self.window_size = window_size or cfg.get("window_size", 500)
        self.degradation_tolerance = cfg.get("degradation_tolerance", 0.05)

        self._y_true: deque[int] = deque(maxlen=self.window_size)
        self._y_pred: deque[int] = deque(maxlen=self.window_size)
        self._history: list[PerformanceSnapshot] = []

        self.baseline_metrics = baseline_metrics or {
            "accuracy": cfg.get("accuracy_threshold", 0.85),
            "f1": cfg.get("f1_threshold", 0.80),
            "precision": cfg.get("precision_threshold", 0.80),
            "recall": cfg.get("recall_threshold", 0.75),
        }

    def record(self, y_true: list[int], y_pred: list[int]) -> None:
        """Add new predictions to the sliding window.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
        """
        for yt, yp in zip(y_true, y_pred):
            self._y_true.append(yt)
            self._y_pred.append(yp)

        logger.debug(
            "performance_recorded",
            n_new=len(y_true),
            window_fill=len(self._y_true),
        )

    def get_current_metrics(self) -> PerformanceSnapshot | None:
        """Compute current metrics over the sliding window.

        Returns:
            PerformanceSnapshot or None if no data recorded.
        """
        if len(self._y_true) == 0:
            return None

        y_true = np.array(list(self._y_true))
        y_pred = np.array(list(self._y_pred))

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            n_samples=len(y_true),
            window_size=self.window_size,
        )

        self._history.append(snapshot)
        return snapshot

    def check_degradation(self) -> list[PerformanceDegradation]:
        """Check if any metric has degraded below baseline thresholds.

        Returns:
            List of PerformanceDegradation records for each metric.
        """
        snapshot = self.get_current_metrics()
        if snapshot is None:
            return []

        results: list[PerformanceDegradation] = []
        current = {
            "accuracy": snapshot.accuracy,
            "f1": snapshot.f1,
            "precision": snapshot.precision,
            "recall": snapshot.recall,
        }

        for metric, current_val in current.items():
            baseline_val = self.baseline_metrics.get(metric, 0.0)
            drop = baseline_val - current_val

            is_degraded = drop > self.degradation_tolerance

            results.append(
                PerformanceDegradation(
                    metric=metric,
                    current_value=current_val,
                    baseline_value=baseline_val,
                    drop=drop,
                    threshold=self.degradation_tolerance,
                    is_degraded=is_degraded,
                )
            )

            if is_degraded:
                logger.warning(
                    "performance_degradation_detected",
                    metric=metric,
                    current=current_val,
                    baseline=baseline_val,
                    drop=drop,
                )

        return results

    def get_history(self, n: int | None = None) -> list[PerformanceSnapshot]:
        """Get performance snapshot history.

        Args:
            n: Number of most recent snapshots to return. None returns all.

        Returns:
            List of PerformanceSnapshot records.
        """
        if n is None:
            return list(self._history)
        return list(self._history[-n:])

    def set_baseline(self, metrics: dict[str, float]) -> None:
        """Update baseline metrics for degradation detection.

        Args:
            metrics: Dictionary of metric name to baseline value.
        """
        self.baseline_metrics.update(metrics)
        logger.info("baseline_updated", metrics=self.baseline_metrics)

    def reset(self) -> None:
        """Clear the sliding window and history."""
        self._y_true.clear()
        self._y_pred.clear()
        self._history.clear()
        logger.info("performance_tracker_reset")
