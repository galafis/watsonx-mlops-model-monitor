"""Shadow deployment that runs a new model alongside production without affecting users."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from src.serving.gateway import ModelGateway

logger = structlog.get_logger(__name__)


@dataclass
class ShadowComparison:
    """Record of production vs shadow prediction divergence."""

    total_requests: int = 0
    agreement_count: int = 0
    divergence_count: int = 0
    divergence_samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        """Fraction of requests where shadow and production agreed."""
        if self.total_requests == 0:
            return 0.0
        return self.agreement_count / self.total_requests

    @property
    def divergence_rate(self) -> float:
        """Fraction of requests where shadow and production disagreed."""
        if self.total_requests == 0:
            return 0.0
        return self.divergence_count / self.total_requests


class ShadowRunner:
    """Run a candidate model in shadow mode alongside production.

    Shadow mode executes the candidate model on the same inputs as production
    but does NOT return its predictions to users. Instead, it records
    agreement/divergence statistics for offline analysis.

    Example:
        >>> runner = ShadowRunner(gateway)
        >>> shadow_pred = runner.run_shadow("canary", [[1.0, 2.0]])
        >>> comparison = runner.get_comparison()
        >>> print(f"Agreement rate: {comparison.agreement_rate:.2%}")
    """

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway
        self._comparison = ShadowComparison()
        self._max_divergence_samples = 100

    def run_shadow(
        self,
        shadow_model_id: str,
        features: list[list[float]] | np.ndarray,
        production_predictions: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Execute inference on the shadow model without affecting users.

        Args:
            shadow_model_id: Model identifier for the shadow/candidate model.
            features: Input feature matrix.
            production_predictions: If provided, compare against these predictions.

        Returns:
            Shadow prediction result, or None if the model is not loaded.
        """
        if not self.gateway.is_loaded(shadow_model_id):
            logger.warning("shadow_model_not_loaded", model_id=shadow_model_id)
            return None

        try:
            shadow_result = self.gateway.predict(shadow_model_id, features)

            if production_predictions is not None:
                self._record_comparison(
                    production_predictions,
                    shadow_result["predictions"],
                    features,
                )

            logger.debug(
                "shadow_inference_completed",
                model_id=shadow_model_id,
                n_samples=len(shadow_result["predictions"]),
            )

            return shadow_result

        except Exception as e:
            logger.error(
                "shadow_inference_failed",
                model_id=shadow_model_id,
                error=str(e),
            )
            return None

    def _record_comparison(
        self,
        production_preds: list[Any],
        shadow_preds: list[Any],
        features: list[list[float]] | np.ndarray,
    ) -> None:
        """Record agreement/divergence between production and shadow predictions."""
        for i, (prod, shadow) in enumerate(zip(production_preds, shadow_preds, strict=False)):
            self._comparison.total_requests += 1
            if prod == shadow:
                self._comparison.agreement_count += 1
            else:
                self._comparison.divergence_count += 1
                if len(self._comparison.divergence_samples) < self._max_divergence_samples:
                    sample = features[i] if isinstance(features, list) else features[i].tolist()
                    self._comparison.divergence_samples.append(
                        {
                            "production": prod,
                            "shadow": shadow,
                            "features": sample,
                        }
                    )

    def get_comparison(self) -> ShadowComparison:
        """Get the current shadow comparison statistics.

        Returns:
            ShadowComparison with agreement and divergence data.
        """
        return self._comparison

    def reset_comparison(self) -> None:
        """Reset comparison statistics for a new evaluation period."""
        self._comparison = ShadowComparison()
        logger.info("shadow_comparison_reset")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of shadow mode statistics.

        Returns:
            Dictionary with key shadow comparison metrics.
        """
        comp = self._comparison
        return {
            "total_requests": comp.total_requests,
            "agreement_rate": comp.agreement_rate,
            "divergence_rate": comp.divergence_rate,
            "divergence_count": comp.divergence_count,
            "sample_divergences": comp.divergence_samples[:5],
        }
