"""Fairness metrics: demographic parity, equalized odds, and calibration across protected groups.

Implements real fairness metric calculations following the definitions from
Fairness and Machine Learning (Barocas, Hardt, Narayanan).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class GroupMetrics:
    """Metrics computed for a single protected group."""

    group_name: str
    group_size: int
    positive_rate: float
    true_positive_rate: float
    false_positive_rate: float
    positive_predictive_value: float
    calibration_score: float


@dataclass
class FairnessResult:
    """Result of fairness evaluation for a single protected attribute."""

    attribute: str
    group_metrics: list[GroupMetrics]
    demographic_parity_difference: float
    demographic_parity_ratio: float
    equalized_odds_difference: float
    calibration_difference: float
    is_fair: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class FairnessReport:
    """Aggregated fairness report across all protected attributes."""

    attribute_results: list[FairnessResult]
    overall_fair: bool
    n_violations: int


class FairnessMonitor:
    """Monitor model fairness across protected groups.

    Computes demographic parity, equalized odds, and calibration metrics
    for each protected attribute. Flags violations when disparities exceed
    configurable thresholds.

    Definitions:
    - Demographic Parity: P(Y_hat=1 | A=a) should be equal across groups.
    - Equalized Odds: P(Y_hat=1 | Y=y, A=a) should be equal across groups for y in {0,1}.
    - Calibration: P(Y=1 | Y_hat=1, A=a) should be equal across groups.

    Example:
        >>> monitor = FairnessMonitor()
        >>> report = monitor.evaluate(
        ...     y_true=labels, y_pred=predictions,
        ...     protected_attributes={"gender": gender_array}
        ... )
        >>> for r in report.attribute_results:
        ...     print(f"{r.attribute}: DP diff={r.demographic_parity_difference:.4f}")
    """

    def __init__(
        self,
        dp_threshold: float | None = None,
        eo_threshold: float | None = None,
        cal_threshold: float | None = None,
    ) -> None:
        cfg = settings.fairness_config
        self.dp_threshold = dp_threshold or cfg.get("demographic_parity_threshold", 0.1)
        self.eo_threshold = eo_threshold or cfg.get("equalized_odds_threshold", 0.1)
        self.cal_threshold = cal_threshold or cfg.get("calibration_threshold", 0.1)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: dict[str, np.ndarray],
    ) -> FairnessReport:
        """Evaluate fairness metrics across all protected attributes.

        Args:
            y_true: Ground truth labels, shape (n_samples,).
            y_pred: Predicted labels, shape (n_samples,).
            protected_attributes: Dictionary mapping attribute names to group arrays.
                Each array should have the same length as y_true with group labels.

        Returns:
            FairnessReport with per-attribute metrics and overall assessment.
        """
        results: list[FairnessResult] = []

        for attr_name, groups in protected_attributes.items():
            result = self._evaluate_attribute(y_true, y_pred, attr_name, groups)
            results.append(result)

        n_violations = sum(len(r.violations) for r in results)
        overall_fair = all(r.is_fair for r in results)

        logger.info(
            "fairness_evaluation_completed",
            n_attributes=len(protected_attributes),
            n_violations=n_violations,
            overall_fair=overall_fair,
        )

        return FairnessReport(
            attribute_results=results,
            overall_fair=overall_fair,
            n_violations=n_violations,
        )

    def _evaluate_attribute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        attr_name: str,
        groups: np.ndarray,
    ) -> FairnessResult:
        """Evaluate fairness for a single protected attribute."""
        unique_groups = np.unique(groups)
        group_metrics_list: list[GroupMetrics] = []

        for group_val in unique_groups:
            mask = groups == group_val
            gm = self._compute_group_metrics(
                y_true[mask],
                y_pred[mask],
                str(group_val),
            )
            group_metrics_list.append(gm)

        dp_diff, dp_ratio = self._demographic_parity(group_metrics_list)
        eo_diff = self._equalized_odds_difference(group_metrics_list)
        cal_diff = self._calibration_difference(group_metrics_list)

        violations: list[str] = []
        if dp_diff > self.dp_threshold:
            violations.append(
                f"Demographic parity violation: difference={dp_diff:.4f} > {self.dp_threshold}"
            )
        if eo_diff > self.eo_threshold:
            violations.append(
                f"Equalized odds violation: difference={eo_diff:.4f} > {self.eo_threshold}"
            )
        if cal_diff > self.cal_threshold:
            violations.append(
                f"Calibration violation: difference={cal_diff:.4f} > {self.cal_threshold}"
            )

        return FairnessResult(
            attribute=attr_name,
            group_metrics=group_metrics_list,
            demographic_parity_difference=dp_diff,
            demographic_parity_ratio=dp_ratio,
            equalized_odds_difference=eo_diff,
            calibration_difference=cal_diff,
            is_fair=len(violations) == 0,
            violations=violations,
        )

    def _compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_name: str,
    ) -> GroupMetrics:
        """Compute classification metrics for a single demographic group."""
        n = len(y_true)
        if n == 0:
            return GroupMetrics(
                group_name=group_name,
                group_size=0,
                positive_rate=0.0,
                true_positive_rate=0.0,
                false_positive_rate=0.0,
                positive_predictive_value=0.0,
                calibration_score=0.0,
            )

        positive_rate = float(np.mean(y_pred == 1))

        # True Positive Rate = TP / (TP + FN) = P(Y_hat=1 | Y=1)
        positives_mask = y_true == 1
        tpr = float(np.mean(y_pred[positives_mask] == 1)) if positives_mask.sum() > 0 else 0.0

        # False Positive Rate = FP / (FP + TN) = P(Y_hat=1 | Y=0)
        negatives_mask = y_true == 0
        fpr = float(np.mean(y_pred[negatives_mask] == 1)) if negatives_mask.sum() > 0 else 0.0

        # Positive Predictive Value = TP / (TP + FP) = P(Y=1 | Y_hat=1)
        predicted_positive_mask = y_pred == 1
        if predicted_positive_mask.sum() > 0:
            ppv = float(np.mean(y_true[predicted_positive_mask] == 1))
        else:
            ppv = 0.0

        return GroupMetrics(
            group_name=group_name,
            group_size=n,
            positive_rate=positive_rate,
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            positive_predictive_value=ppv,
            calibration_score=ppv,
        )

    def _demographic_parity(self, group_metrics: list[GroupMetrics]) -> tuple[float, float]:
        """Compute demographic parity difference and ratio.

        Demographic parity requires P(Y_hat=1 | A=a) to be equal for all groups.

        Returns:
            (max_difference, min_ratio) of positive rates across groups.
        """
        rates = [gm.positive_rate for gm in group_metrics if gm.group_size > 0]
        if len(rates) < 2:
            return 0.0, 1.0

        max_diff = float(max(rates) - min(rates))
        max_rate = max(rates)
        min_ratio = float(min(rates) / max_rate) if max_rate > 0 else 1.0

        return max_diff, min_ratio

    def _equalized_odds_difference(self, group_metrics: list[GroupMetrics]) -> float:
        """Compute equalized odds difference.

        Equalized odds requires both TPR and FPR to be equal across groups.
        Returns the maximum of |TPR_a - TPR_b| and |FPR_a - FPR_b| over all pairs.
        """
        valid = [gm for gm in group_metrics if gm.group_size > 0]
        if len(valid) < 2:
            return 0.0

        tprs = [gm.true_positive_rate for gm in valid]
        fprs = [gm.false_positive_rate for gm in valid]

        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)

        return float(max(tpr_diff, fpr_diff))

    def _calibration_difference(self, group_metrics: list[GroupMetrics]) -> float:
        """Compute calibration difference across groups.

        Calibration requires P(Y=1 | Y_hat=1, A=a) to be equal across groups.
        """
        valid = [gm for gm in group_metrics if gm.group_size > 0]
        if len(valid) < 2:
            return 0.0

        ppvs = [gm.positive_predictive_value for gm in valid]
        return float(max(ppvs) - min(ppvs))
