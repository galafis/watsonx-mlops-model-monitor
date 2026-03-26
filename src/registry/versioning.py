"""Model version comparison and promotion logic for A/B testing and canary rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.config import settings
from src.registry.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)


@dataclass
class VersionComparison:
    """Result of comparing two model versions."""

    candidate_version: int
    production_version: int
    candidate_metrics: dict[str, float]
    production_metrics: dict[str, float]
    improvements: dict[str, float]
    should_promote: bool
    reason: str


class VersionManager:
    """Compare model versions and manage promotion decisions.

    Implements logic for deciding whether a candidate model should replace
    the current production model based on metric improvements and thresholds.

    Example:
        >>> manager = VersionManager()
        >>> comparison = manager.compare_versions("credit_risk", candidate=3, production=2)
        >>> if comparison.should_promote:
        ...     manager.promote_candidate("credit_risk", version=3)
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self.min_samples = settings.deployment.get("min_samples_before_promote", 500)

    def compare_versions(
        self,
        model_name: str,
        candidate_version: int,
        production_version: int,
        primary_metric: str = "f1",
        min_improvement: float = 0.01,
    ) -> VersionComparison:
        """Compare candidate model against production model metrics.

        Args:
            model_name: Registered model name.
            candidate_version: Version number of the candidate.
            production_version: Version number of the current production model.
            primary_metric: Main metric for promotion decision.
            min_improvement: Minimum improvement required to recommend promotion.

        Returns:
            VersionComparison with detailed comparison results.
        """
        from src.training.experiment import ExperimentTracker

        tracker = ExperimentTracker()

        versions = self.registry.list_versions(model_name)
        candidate = next((v for v in versions if v.version == candidate_version), None)
        production = next((v for v in versions if v.version == production_version), None)

        if candidate is None or production is None:
            raise ValueError("One or both model versions not found in registry.")

        candidate_metrics = tracker.get_run_metrics(candidate.run_id)
        production_metrics = tracker.get_run_metrics(production.run_id)

        improvements = {
            k: candidate_metrics.get(k, 0) - production_metrics.get(k, 0)
            for k in set(candidate_metrics) | set(production_metrics)
        }

        primary_improvement = improvements.get(primary_metric, 0)
        should_promote = primary_improvement >= min_improvement

        reason = (
            f"{primary_metric} improved by {primary_improvement:.4f} "
            f"(>= {min_improvement} threshold)"
            if should_promote
            else f"{primary_metric} improvement {primary_improvement:.4f} "
            f"below {min_improvement} threshold"
        )

        logger.info(
            "version_comparison_completed",
            model=model_name,
            candidate=candidate_version,
            production=production_version,
            improvement=primary_improvement,
            should_promote=should_promote,
        )

        return VersionComparison(
            candidate_version=candidate_version,
            production_version=production_version,
            candidate_metrics=candidate_metrics,
            production_metrics=production_metrics,
            improvements=improvements,
            should_promote=should_promote,
            reason=reason,
        )

    def promote_candidate(
        self,
        model_name: str,
        version: int,
    ) -> str:
        """Promote a candidate model version to Production stage.

        Args:
            model_name: Registered model name.
            version: Version to promote.

        Returns:
            The new stage string ('Production').
        """
        new_stage = self.registry.transition_stage(
            model_name=model_name,
            version=version,
            stage="Production",
            archive_existing=True,
        )

        logger.info(
            "candidate_promoted",
            model=model_name,
            version=version,
            stage=new_stage,
        )

        return new_stage

    def rollback(self, model_name: str, to_version: int) -> str:
        """Rollback production to a previous model version.

        Args:
            model_name: Registered model name.
            to_version: Version to rollback to.

        Returns:
            The new stage string.
        """
        logger.warning(
            "model_rollback_initiated",
            model=model_name,
            rollback_to=to_version,
        )

        return self.registry.transition_stage(
            model_name=model_name,
            version=to_version,
            stage="Production",
            archive_existing=True,
        )
