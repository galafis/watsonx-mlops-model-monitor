"""Tests for the statistical drift detection module."""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift_detector import (
    DriftDetector,
    DriftReport,
    DriftSeverity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector() -> DriftDetector:
    """Return a DriftDetector with default thresholds."""
    return DriftDetector(psi_threshold=0.2, ks_alpha=0.05, kl_threshold=0.1, js_threshold=0.1)


@pytest.fixture()
def reference_data() -> np.ndarray:
    """Stable reference distribution (N=1000, 3 features)."""
    rng = np.random.RandomState(42)
    return rng.normal(loc=0.0, scale=1.0, size=(1000, 3))


@pytest.fixture()
def same_distribution_data() -> np.ndarray:
    """Production data drawn from the same distribution as reference."""
    rng = np.random.RandomState(99)
    return rng.normal(loc=0.0, scale=1.0, size=(1000, 3))


@pytest.fixture()
def shifted_data() -> np.ndarray:
    """Production data with significant distribution shift."""
    rng = np.random.RandomState(99)
    return rng.normal(loc=3.0, scale=2.0, size=(1000, 3))


# ---------------------------------------------------------------------------
# DriftDetector.detect — full report
# ---------------------------------------------------------------------------


class TestDriftDetectorNoShift:
    """When reference and production come from the same distribution, no drift is expected."""

    def test_no_drift_detected(
        self,
        detector: DriftDetector,
        reference_data: np.ndarray,
        same_distribution_data: np.ndarray,
    ) -> None:
        report = detector.detect(reference_data, same_distribution_data)
        assert isinstance(report, DriftReport)
        assert report.overall_drifted is False
        assert report.n_drifted_features == 0

    def test_drift_ratio_is_zero(
        self,
        detector: DriftDetector,
        reference_data: np.ndarray,
        same_distribution_data: np.ndarray,
    ) -> None:
        report = detector.detect(reference_data, same_distribution_data)
        assert report.drift_ratio == pytest.approx(0.0)

    def test_feature_count_matches(
        self,
        detector: DriftDetector,
        reference_data: np.ndarray,
        same_distribution_data: np.ndarray,
    ) -> None:
        report = detector.detect(reference_data, same_distribution_data)
        assert report.n_total_features == 3
        assert len(report.feature_results) == 3

    def test_psi_is_low(
        self,
        detector: DriftDetector,
        reference_data: np.ndarray,
        same_distribution_data: np.ndarray,
    ) -> None:
        report = detector.detect(reference_data, same_distribution_data)
        for r in report.feature_results:
            assert r.psi < 0.2, f"PSI should be low for stable distribution: got {r.psi}"

    def test_severity_none_or_low(
        self,
        detector: DriftDetector,
        reference_data: np.ndarray,
        same_distribution_data: np.ndarray,
    ) -> None:
        report = detector.detect(reference_data, same_distribution_data)
        for r in report.feature_results:
            assert r.severity in (DriftSeverity.NONE, DriftSeverity.LOW)


class TestDriftDetectorWithShift:
    """When production is significantly shifted, drift must be detected."""

    def test_drift_detected(
        self, detector: DriftDetector, reference_data: np.ndarray, shifted_data: np.ndarray
    ) -> None:
        report = detector.detect(reference_data, shifted_data)
        assert report.overall_drifted is True
        assert report.n_drifted_features > 0

    def test_all_features_drifted(
        self, detector: DriftDetector, reference_data: np.ndarray, shifted_data: np.ndarray
    ) -> None:
        report = detector.detect(reference_data, shifted_data)
        assert report.n_drifted_features == 3

    def test_psi_exceeds_threshold(
        self, detector: DriftDetector, reference_data: np.ndarray, shifted_data: np.ndarray
    ) -> None:
        report = detector.detect(reference_data, shifted_data)
        for r in report.feature_results:
            assert r.psi > 0.2

    def test_severity_moderate_or_high(
        self, detector: DriftDetector, reference_data: np.ndarray, shifted_data: np.ndarray
    ) -> None:
        report = detector.detect(reference_data, shifted_data)
        for r in report.feature_results:
            assert r.severity in (DriftSeverity.MODERATE, DriftSeverity.HIGH)

    def test_ks_pvalue_significant(
        self, detector: DriftDetector, reference_data: np.ndarray, shifted_data: np.ndarray
    ) -> None:
        report = detector.detect(reference_data, shifted_data)
        for r in report.feature_results:
            assert r.ks_pvalue < 0.05


# ---------------------------------------------------------------------------
# Individual statistical tests
# ---------------------------------------------------------------------------


class TestPSIComputation:
    """Unit tests for PSI calculation."""

    def test_psi_identical_distributions(self, detector: DriftDetector) -> None:
        data = np.random.RandomState(42).normal(0, 1, 500)
        psi = detector._compute_psi(data, data)
        assert psi >= 0.0
        assert psi < 0.05

    def test_psi_is_non_negative(self, detector: DriftDetector) -> None:
        ref = np.random.RandomState(42).normal(0, 1, 500)
        prod = np.random.RandomState(99).normal(5, 2, 500)
        psi = detector._compute_psi(ref, prod)
        assert psi >= 0.0

    def test_psi_large_shift(self, detector: DriftDetector) -> None:
        ref = np.random.RandomState(42).normal(0, 1, 500)
        prod = np.random.RandomState(99).normal(10, 1, 500)
        psi = detector._compute_psi(ref, prod)
        assert psi > 0.2


class TestKSTest:
    """Unit tests for Kolmogorov-Smirnov test."""

    def test_same_distribution_high_pvalue(self, detector: DriftDetector) -> None:
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        _stat, pval = detector._compute_ks_test(a, b)
        assert pval > 0.05

    def test_different_distribution_low_pvalue(self, detector: DriftDetector) -> None:
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(5, 1, 500)
        stat, pval = detector._compute_ks_test(a, b)
        assert pval < 0.05
        assert stat > 0.0


class TestJSDistance:
    """Unit tests for Jensen-Shannon distance."""

    def test_js_distance_range(self, detector: DriftDetector) -> None:
        ref = np.random.RandomState(42).normal(0, 1, 500)
        prod = np.random.RandomState(99).normal(0, 1, 500)
        js = detector._compute_js_distance(ref, prod)
        assert 0.0 <= js <= 1.0

    def test_js_large_shift(self, detector: DriftDetector) -> None:
        ref = np.random.RandomState(42).normal(0, 1, 500)
        prod = np.random.RandomState(99).normal(10, 1, 500)
        js = detector._compute_js_distance(ref, prod)
        assert js > 0.1


class TestKLDivergence:
    """Unit tests for KL-divergence."""

    def test_kl_non_negative(self, detector: DriftDetector) -> None:
        ref = np.random.RandomState(42).normal(0, 1, 500)
        prod = np.random.RandomState(99).normal(0, 1, 500)
        kl = detector._compute_kl_divergence(ref, prod)
        assert kl >= 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the drift detector."""

    def test_1d_input(self, detector: DriftDetector) -> None:
        ref = np.random.RandomState(42).normal(0, 1, 500)
        prod = np.random.RandomState(99).normal(0, 1, 500)
        report = detector.detect(ref, prod)
        assert report.n_total_features == 1

    def test_custom_feature_names(self, detector: DriftDetector) -> None:
        ref = np.random.RandomState(42).normal(0, 1, (500, 2))
        prod = np.random.RandomState(99).normal(0, 1, (500, 2))
        report = detector.detect(ref, prod, feature_names=["age", "income"])
        assert report.feature_results[0].feature_name == "age"
        assert report.feature_results[1].feature_name == "income"

    def test_default_feature_names(self, detector: DriftDetector) -> None:
        ref = np.random.RandomState(42).normal(0, 1, (500, 2))
        prod = np.random.RandomState(99).normal(0, 1, (500, 2))
        report = detector.detect(ref, prod)
        assert report.feature_results[0].feature_name == "feature_0"

    def test_drift_result_has_details(
        self, detector: DriftDetector, reference_data: np.ndarray, shifted_data: np.ndarray
    ) -> None:
        report = detector.detect(reference_data, shifted_data)
        r = report.feature_results[0]
        assert "reference_mean" in r.details
        assert "production_mean" in r.details
        assert "reference_std" in r.details
        assert "production_std" in r.details


class TestDriftSeverityClassification:
    """Tests for the severity classification method."""

    def test_none_severity(self, detector: DriftDetector) -> None:
        severity = detector._classify_severity(psi=0.05, ks_pvalue=0.5, js=0.03)
        assert severity == DriftSeverity.NONE

    def test_low_severity(self, detector: DriftDetector) -> None:
        severity = detector._classify_severity(psi=0.12, ks_pvalue=0.5, js=0.06)
        assert severity == DriftSeverity.LOW

    def test_moderate_severity(self, detector: DriftDetector) -> None:
        severity = detector._classify_severity(psi=0.25, ks_pvalue=0.01, js=0.12)
        assert severity == DriftSeverity.MODERATE

    def test_high_severity(self, detector: DriftDetector) -> None:
        severity = detector._classify_severity(psi=0.5, ks_pvalue=0.001, js=0.3)
        assert severity == DriftSeverity.HIGH
