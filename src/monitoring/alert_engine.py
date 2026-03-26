"""Alert engine with configurable thresholds, cooldowns, and notification channels."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A single alert record."""

    alert_id: str
    severity: AlertSeverity
    category: str
    title: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    acknowledged: bool = False

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class AlertEngine:
    """Configurable alert engine for model monitoring notifications.

    Manages alert creation, deduplication via cooldowns, history tracking,
    and dispatching to configured notification channels (log, prometheus).

    Example:
        >>> engine = AlertEngine()
        >>> engine.fire(
        ...     severity=AlertSeverity.WARNING,
        ...     category="drift",
        ...     title="Feature drift detected",
        ...     message="PSI=0.35 for feature 'age' exceeds threshold 0.2",
        ... )
    """

    def __init__(self, cooldown_seconds: int | None = None) -> None:
        cfg = settings.alerting
        self.cooldown_seconds = cooldown_seconds or cfg.get("cooldown_seconds", 300)
        self.enabled = cfg.get("enabled", True)
        self._alerts: list[Alert] = []
        self._last_fired: dict[str, float] = {}
        self._alert_counter = 0

    def fire(
        self,
        severity: AlertSeverity,
        category: str,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Fire an alert if not in cooldown period.

        Args:
            severity: Alert severity level.
            category: Category (e.g., 'drift', 'fairness', 'performance').
            title: Short alert title.
            message: Detailed alert message.
            details: Optional additional data.

        Returns:
            The Alert object if fired, None if suppressed by cooldown.
        """
        if not self.enabled:
            return None

        cooldown_key = f"{category}:{title}"
        now = time.time()

        last = self._last_fired.get(cooldown_key, 0)
        if now - last < self.cooldown_seconds:
            logger.debug(
                "alert_suppressed_cooldown",
                category=category,
                title=title,
                remaining=self.cooldown_seconds - (now - last),
            )
            return None

        self._alert_counter += 1
        alert = Alert(
            alert_id=f"ALT-{self._alert_counter:06d}",
            severity=severity,
            category=category,
            title=title,
            message=message,
            details=details or {},
        )

        self._alerts.append(alert)
        self._last_fired[cooldown_key] = now

        self._dispatch(alert)
        return alert

    def _dispatch(self, alert: Alert) -> None:
        """Dispatch alert to configured channels."""
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
        }.get(alert.severity, logger.info)

        log_method(
            "alert_fired",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
            category=alert.category,
            title=alert.title,
            message=alert.message,
        )

    def get_history(
        self,
        category: str | None = None,
        severity: AlertSeverity | None = None,
        limit: int = 50,
    ) -> list[Alert]:
        """Retrieve alert history with optional filtering.

        Args:
            category: Filter by alert category.
            severity: Filter by severity level.
            limit: Maximum number of alerts to return.

        Returns:
            List of Alert objects, most recent first.
        """
        filtered = self._alerts
        if category:
            filtered = [a for a in filtered if a.category == category]
        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        return list(reversed(filtered[-limit:]))

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged.

        Args:
            alert_id: The alert ID to acknowledge.

        Returns:
            True if found and acknowledged, False otherwise.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info("alert_acknowledged", alert_id=alert_id)
                return True
        return False

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of the alert engine.

        Returns:
            Dictionary with total counts by severity and category.
        """
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        unacknowledged = 0

        for alert in self._alerts:
            by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1
            by_category[alert.category] = by_category.get(alert.category, 0) + 1
            if not alert.acknowledged:
                unacknowledged += 1

        return {
            "total_alerts": len(self._alerts),
            "unacknowledged": unacknowledged,
            "by_severity": by_severity,
            "by_category": by_category,
        }

    def clear_history(self) -> None:
        """Clear all alerts and reset cooldowns."""
        self._alerts.clear()
        self._last_fired.clear()
        self._alert_counter = 0
        logger.info("alert_history_cleared")
