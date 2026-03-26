"""Tests for the fairness monitoring module."""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.fairness_monitor import (
    FairnessMonitor,
    FairnessReport,
    GroupMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def monitor() -> FairnessMonitor:
    """Return a FairnessMonitor with default thresholds."""
    return FairnessMonitor(dp_threshold=0.1, eo_threshold=0.1, cal_threshold=0.1)


@pytest.fixture()
def fair_scenario() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Scenario where model treats groups equally (fair)."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.randint(0, 2, n)
    # Predictions are identical to truth (perfect model, same for both groups)
    y_pred = y_true.copy()
    gender = np.array(["M"] * 100 + ["F"] * 100)
    return y_true, y_pred, {"gender": gender}


@pytest.fixture()
def unfair_scenario() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Scenario with significant disparity between groups (unfair)."""
    # Group A: mostly positive predictions
    y_true_a = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred_a = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # Group B: mostly negative predictions (biased)
    y_true_b = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred_b = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])

    y_true = np.concatenate([y_true_a, y_true_b])
    y_pred = np.concatenate([y_pred_a, y_pred_b])
    groups = np.array(["A"] * 10 + ["B"] * 10)

    return y_true, y_pred, {"group": groups}


# ---------------------------------------------------------------------------
# Fair scenario tests
# ---------------------------------------------------------------------------


class TestFairScenario:
    """When the model treats groups equally, no violations should be raised."""

    def test_overall_fair(
        self,
        monitor: FairnessMonitor,
        fair_scenario: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        y_true, y_pred, protected = fair_scenario
        report = monitor.evaluate(y_true, y_pred, protected)
        assert isinstance(report, FairnessReport)
        assert report.overall_fair is True

    def test_no_violations(
        self,
        monitor: FairnessMonitor,
        fair_scenario: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        y_true, y_pred, protected = fair_scenario
        report = monitor.evaluate(y_true, y_pred, protected)
        assert report.n_violations == 0

    def test_low_demographic_parity_diff(
        self,
        monitor: FairnessMonitor,
        fair_scenario: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        y_true, y_pred, protected = fair_scenario
        report = monitor.evaluate(y_true, y_pred, protected)
        for result in report.attribute_results:
            assert result.demographic_parity_difference <= 0.1


# ---------------------------------------------------------------------------
# Unfair scenario tests
# ---------------------------------------------------------------------------


class TestUnfairScenario:
    """When the model is biased, violations must be detected."""

    def test_not_fair(
        self,
        monitor: FairnessMonitor,
        unfair_scenario: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        y_true, y_pred, protected = unfair_scenario
        report = monitor.evaluate(y_true, y_pred, protected)
        assert report.overall_fair is False

    def test_violations_detected(
        self,
        monitor: FairnessMonitor,
        unfair_scenario: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        y_true, y_pred, protected = unfair_scenario
        report = monitor.evaluate(y_true, y_pred, protected)
        assert report.n_violations > 0

    def test_demographic_parity_violation(
        self,
        monitor: FairnessMonitor,
        unfair_scenario: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        y_true, y_pred, protected = unfair_scenario
        report = monitor.evaluate(y_true, y_pred, protected)
        result = report.attribute_results[0]
        assert result.demographic_parity_difference > 0.1

    def test_equalized_odds_violation(
        self,
        monitor: FairnessMonitor,
        unfair_scenario: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        y_true, y_pred, protected = unfair_scenario
        report = monitor.evaluate(y_true, y_pred, protected)
        result = report.attribute_results[0]
        assert result.equalized_odds_difference > 0.1


# ---------------------------------------------------------------------------
# GroupMetrics computation
# ---------------------------------------------------------------------------


class TestGroupMetrics:
    """Test individual group metric calculations."""

    def test_perfect_predictions(self, monitor: FairnessMonitor) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        gm = monitor._compute_group_metrics(y_true, y_pred, "test")
        assert gm.group_name == "test"
        assert gm.group_size == 4
        assert gm.positive_rate == 0.5
        assert gm.true_positive_rate == 1.0
        assert gm.false_positive_rate == 0.0
        assert gm.positive_predictive_value == 1.0

    def test_all_negative_predictions(self, monitor: FairnessMonitor) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        gm = monitor._compute_group_metrics(y_true, y_pred, "test")
        assert gm.positive_rate == 0.0
        assert gm.true_positive_rate == 0.0
        assert gm.false_positive_rate == 0.0
        assert gm.positive_predictive_value == 0.0

    def test_all_positive_predictions(self, monitor: FairnessMonitor) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        gm = monitor._compute_group_metrics(y_true, y_pred, "test")
        assert gm.positive_rate == 1.0
        assert gm.true_positive_rate == 1.0
        assert gm.false_positive_rate == 1.0
        assert gm.positive_predictive_value == 0.5

    def test_empty_group(self, monitor: FairnessMonitor) -> None:
        y_true = np.array([])
        y_pred = np.array([])
        gm = monitor._compute_group_metrics(y_true, y_pred, "empty")
        assert gm.group_size == 0
        assert gm.positive_rate == 0.0


# ---------------------------------------------------------------------------
# Metric computation methods
# ---------------------------------------------------------------------------


class TestDemographicParity:
    """Tests for demographic parity calculation."""

    def test_equal_rates(self, monitor: FairnessMonitor) -> None:
        groups = [
            GroupMetrics("A", 50, 0.5, 0.8, 0.1, 0.9, 0.9),
            GroupMetrics("B", 50, 0.5, 0.8, 0.1, 0.9, 0.9),
        ]
        dp_diff, dp_ratio = monitor._demographic_parity(groups)
        assert dp_diff == pytest.approx(0.0)
        assert dp_ratio == pytest.approx(1.0)

    def test_unequal_rates(self, monitor: FairnessMonitor) -> None:
        groups = [
            GroupMetrics("A", 50, 0.8, 0.8, 0.1, 0.9, 0.9),
            GroupMetrics("B", 50, 0.4, 0.8, 0.1, 0.9, 0.9),
        ]
        dp_diff, dp_ratio = monitor._demographic_parity(groups)
        assert dp_diff == pytest.approx(0.4)
        assert dp_ratio == pytest.approx(0.5)

    def test_single_group(self, monitor: FairnessMonitor) -> None:
        groups = [GroupMetrics("A", 50, 0.5, 0.8, 0.1, 0.9, 0.9)]
        dp_diff, dp_ratio = monitor._demographic_parity(groups)
        assert dp_diff == 0.0
        assert dp_ratio == 1.0


class TestEqualizedOdds:
    """Tests for equalized odds difference."""

    def test_equal_tpr_fpr(self, monitor: FairnessMonitor) -> None:
        groups = [
            GroupMetrics("A", 50, 0.5, 0.8, 0.1, 0.9, 0.9),
            GroupMetrics("B", 50, 0.5, 0.8, 0.1, 0.9, 0.9),
        ]
        eo = monitor._equalized_odds_difference(groups)
        assert eo == pytest.approx(0.0)

    def test_unequal_tpr(self, monitor: FairnessMonitor) -> None:
        groups = [
            GroupMetrics("A", 50, 0.5, 0.9, 0.1, 0.9, 0.9),
            GroupMetrics("B", 50, 0.5, 0.5, 0.1, 0.9, 0.9),
        ]
        eo = monitor._equalized_odds_difference(groups)
        assert eo == pytest.approx(0.4)


class TestCalibrationDifference:
    """Tests for calibration difference."""

    def test_equal_calibration(self, monitor: FairnessMonitor) -> None:
        groups = [
            GroupMetrics("A", 50, 0.5, 0.8, 0.1, 0.85, 0.85),
            GroupMetrics("B", 50, 0.5, 0.8, 0.1, 0.85, 0.85),
        ]
        cal = monitor._calibration_difference(groups)
        assert cal == pytest.approx(0.0)

    def test_unequal_calibration(self, monitor: FairnessMonitor) -> None:
        groups = [
            GroupMetrics("A", 50, 0.5, 0.8, 0.1, 0.9, 0.9),
            GroupMetrics("B", 50, 0.5, 0.8, 0.1, 0.6, 0.6),
        ]
        cal = monitor._calibration_difference(groups)
        assert cal == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Multiple protected attributes
# ---------------------------------------------------------------------------


class TestMultipleAttributes:
    """Test evaluation with multiple protected attributes."""

    def test_multiple_attributes(self, monitor: FairnessMonitor) -> None:
        n = 100
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, n)
        y_pred = y_true.copy()

        protected = {
            "gender": np.array(["M"] * 50 + ["F"] * 50),
            "age_group": np.array(["young"] * 50 + ["old"] * 50),
        }

        report = monitor.evaluate(y_true, y_pred, protected)
        assert len(report.attribute_results) == 2
        assert report.attribute_results[0].attribute == "gender"
        assert report.attribute_results[1].attribute == "age_group"
