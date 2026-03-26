"""A/B test routing with canary and blue-green deployment strategies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.config import settings
from src.serving.gateway import ModelGateway

logger = structlog.get_logger(__name__)


class RoutingStrategy(str, Enum):
    """Available traffic routing strategies."""

    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    SHADOW = "shadow"
    DIRECT = "direct"


@dataclass
class RoutingConfig:
    """Configuration for traffic routing between model versions."""

    strategy: RoutingStrategy = RoutingStrategy.CANARY
    canary_weight: float = 0.1
    primary_model: str = "production"
    candidate_model: str = "canary"


@dataclass
class RoutingResult:
    """Result of a routed prediction including which model was used."""

    prediction: dict[str, Any]
    routed_to: str
    strategy: str
    shadow_prediction: dict[str, Any] | None = None


class ABRouter:
    """Traffic router for A/B testing, canary rollouts, and blue-green deployments.

    Routes inference requests between production and candidate models
    according to the configured strategy and traffic split weights.

    Example:
        >>> router = ABRouter(gateway, config=RoutingConfig(strategy=RoutingStrategy.CANARY))
        >>> result = router.route([[1.0, 2.0, 3.0]])
        >>> print(result.routed_to)
    """

    def __init__(
        self,
        gateway: ModelGateway,
        config: RoutingConfig | None = None,
    ) -> None:
        self.gateway = gateway
        deploy_cfg = settings.deployment

        if config:
            self.config = config
        else:
            strategy_str = deploy_cfg.get("strategy", "canary")
            self.config = RoutingConfig(
                strategy=RoutingStrategy(strategy_str),
                canary_weight=deploy_cfg.get("canary_weight", 0.1),
            )

        self._request_count = 0
        self._routing_stats: dict[str, int] = {"production": 0, "canary": 0}

    def route(self, features: list[list[float]] | np.ndarray) -> RoutingResult:
        """Route a prediction request according to the configured strategy.

        Args:
            features: Input feature matrix.

        Returns:
            RoutingResult with the prediction and routing metadata.
        """
        self._request_count += 1

        if self.config.strategy == RoutingStrategy.CANARY:
            return self._route_canary(features)
        elif self.config.strategy == RoutingStrategy.BLUE_GREEN:
            return self._route_blue_green(features)
        elif self.config.strategy == RoutingStrategy.SHADOW:
            return self._route_shadow(features)
        else:
            return self._route_direct(features)

    def _route_canary(self, features: list[list[float]] | np.ndarray) -> RoutingResult:
        """Send a fraction of traffic to the canary model."""
        use_canary = random.random() < self.config.canary_weight

        if use_canary and self.gateway.is_loaded(self.config.candidate_model):
            target = self.config.candidate_model
        else:
            target = self.config.primary_model

        prediction = self.gateway.predict(target, features)
        self._routing_stats[target] = self._routing_stats.get(target, 0) + 1

        logger.debug(
            "canary_routing",
            target=target,
            canary_weight=self.config.canary_weight,
        )

        return RoutingResult(
            prediction=prediction,
            routed_to=target,
            strategy="canary",
        )

    def _route_blue_green(self, features: list[list[float]] | np.ndarray) -> RoutingResult:
        """All-or-nothing routing to the active environment (blue or green)."""
        target = self.config.primary_model
        prediction = self.gateway.predict(target, features)

        return RoutingResult(
            prediction=prediction,
            routed_to=target,
            strategy="blue_green",
        )

    def _route_shadow(self, features: list[list[float]] | np.ndarray) -> RoutingResult:
        """Route to production but also run candidate in shadow mode."""
        from src.serving.shadow_mode import ShadowRunner

        production_prediction = self.gateway.predict(self.config.primary_model, features)

        shadow_prediction = None
        if self.gateway.is_loaded(self.config.candidate_model):
            runner = ShadowRunner(self.gateway)
            shadow_prediction = runner.run_shadow(
                self.config.candidate_model, features
            )

        return RoutingResult(
            prediction=production_prediction,
            routed_to=self.config.primary_model,
            strategy="shadow",
            shadow_prediction=shadow_prediction,
        )

    def _route_direct(self, features: list[list[float]] | np.ndarray) -> RoutingResult:
        """Route all traffic to the primary model."""
        prediction = self.gateway.predict(self.config.primary_model, features)

        return RoutingResult(
            prediction=prediction,
            routed_to=self.config.primary_model,
            strategy="direct",
        )

    def get_stats(self) -> dict[str, Any]:
        """Return routing statistics.

        Returns:
            Dictionary with total requests and per-model counts.
        """
        return {
            "total_requests": self._request_count,
            "routing_stats": dict(self._routing_stats),
            "strategy": self.config.strategy.value,
            "canary_weight": self.config.canary_weight,
        }

    def update_canary_weight(self, weight: float) -> None:
        """Update the canary traffic percentage.

        Args:
            weight: New weight between 0.0 and 1.0.
        """
        self.config.canary_weight = max(0.0, min(1.0, weight))
        logger.info("canary_weight_updated", new_weight=self.config.canary_weight)

    def switch_blue_green(self) -> str:
        """Swap primary and candidate models for blue-green deployment.

        Returns:
            The new primary model identifier.
        """
        self.config.primary_model, self.config.candidate_model = (
            self.config.candidate_model,
            self.config.primary_model,
        )
        logger.info(
            "blue_green_switched",
            new_primary=self.config.primary_model,
            new_candidate=self.config.candidate_model,
        )
        return self.config.primary_model
