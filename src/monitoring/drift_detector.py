"""Statistical drift detection using PSI, KL-divergence, JS-distance, and KS-test.

Implements production-grade data drift detection comparing reference (training)
distributions against live production data using scipy.stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

_EPSILON = 1e-10


class DriftSeverity(str, Enum):
    """Severity classification for drift detection results."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""

    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    kl_divergence: float
    js_distance: float
    is_drifted: bool
    severity: DriftSeverity
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Aggregated drift report across all features."""

    feature_results: list[DriftResult]
    overall_drifted: bool
    n_drifted_features: int
    n_total_features: int
    timestamp: str = ""

    @property
    def drift_ratio(self) -> float:
        """Fraction of features that are drifted."""
        if self.n_total_features == 0:
            return 0.0
        return self.n_drifted_features / self.n_total_features


class DriftDetector:
    """Statistical drift detector comparing reference and production distributions.

    Uses four complementary statistical tests:
    - PSI (Population Stability Index): binned distribution shift measure
    - KL-divergence: asymmetric information-theoretic divergence
    - JS-distance: symmetric variant of KL via Jensen-Shannon
    - KS-test: non-parametric test for distribution equality (scipy.stats.ks_2samp)

    Example:
        >>> detector = DriftDetector()
        >>> report = detector.detect(reference_data, production_data, feature_names)
        >>> for r in report.feature_results:
        ...     if r.is_drifted:
        ...         print(f"{r.feature_name}: PSI={r.psi:.4f}, severity={r.severity}")
    """

    def __init__(
        self,
        psi_threshold: float | None = None,
        ks_alpha: float | None = None,
        kl_threshold: float | None = None,
        js_threshold: float | None = None,
        n_bins: int = 10,
    ) -> None:
        cfg = settings.drift_config
        self.psi_threshold = psi_threshold or cfg.get("psi_threshold", 0.2)
        self.ks_alpha = ks_alpha or cfg.get("ks_alpha", 0.05)
        self.kl_threshold = kl_threshold or cfg.get("kl_threshold", 0.1)
        self.js_threshold = js_threshold or cfg.get("js_threshold", 0.1)
        self.n_bins = n_bins

    def detect(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DriftReport:
        """Run drift detection across all features.

        Args:
            reference: Reference (training) data, shape (n_samples, n_features).
            production: Production data, shape (n_samples, n_features).
            feature_names: Optional feature name labels.

        Returns:
            DriftReport with per-feature results and overall drift assessment.
        """
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if production.ndim == 1:
            production = production.reshape(-1, 1)

        n_features = reference.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        results: list[DriftResult] = []
        for i in range(n_features):
            result = self._detect_single_feature(
                reference[:, i],
                production[:, i],
                feature_names[i],
            )
            results.append(result)

        n_drifted = sum(1 for r in results if r.is_drifted)
        overall = n_drifted > 0

        logger.info(
            "drift_detection_completed",
            n_features=n_features,
            n_drifted=n_drifted,
            overall_drifted=overall,
        )

        return DriftReport(
            feature_results=results,
            overall_drifted=overall,
            n_drifted_features=n_drifted,
            n_total_features=n_features,
        )

    def _detect_single_feature(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_name: str,
    ) -> DriftResult:
        """Run all drift tests on a single feature."""
        psi = self._compute_psi(reference, production)
        ks_stat, ks_pval = self._compute_ks_test(reference, production)
        kl = self._compute_kl_divergence(reference, production)
        js = self._compute_js_distance(reference, production)

        is_drifted = (
            psi > self.psi_threshold
            or ks_pval < self.ks_alpha
            or js > self.js_threshold
        )

        severity = self._classify_severity(psi, ks_pval, js)

        return DriftResult(
            feature_name=feature_name,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            kl_divergence=kl,
            js_distance=js,
            is_drifted=is_drifted,
            severity=severity,
            details={
                "reference_mean": float(np.mean(reference)),
                "production_mean": float(np.mean(production)),
                "reference_std": float(np.std(reference)),
                "production_std": float(np.std(production)),
            },
        )

    def _compute_psi(self, reference: np.ndarray, production: np.ndarray) -> float:
        """Compute Population Stability Index between two distributions.

        PSI measures the shift in distribution of a variable between two samples.
        Values: < 0.1 = no shift, 0.1-0.2 = moderate, > 0.2 = significant.

        Uses equal-width binning based on reference distribution quantiles.
        """
        breakpoints = np.percentile(
            reference,
            np.linspace(0, 100, self.n_bins + 1),
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
        prod_counts = np.histogram(production, bins=breakpoints)[0].astype(float)

        ref_pct = ref_counts / ref_counts.sum() + _EPSILON
        prod_pct = prod_counts / prod_counts.sum() + _EPSILON

        psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return max(psi, 0.0)

    def _compute_ks_test(
        self, reference: np.ndarray, production: np.ndarray
    ) -> tuple[float, float]:
        """Run two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp).

        Tests the null hypothesis that both samples are drawn from the same
        continuous distribution. Returns the test statistic and p-value.
        """
        stat, pvalue = stats.ks_2samp(reference, production)
        return float(stat), float(pvalue)

    def _compute_kl_divergence(
        self, reference: np.ndarray, production: np.ndarray
    ) -> float:
        """Compute KL-divergence D_KL(production || reference).

        Uses histogram-based probability estimation with Laplace smoothing.
        """
        breakpoints = np.percentile(
            reference,
            np.linspace(0, 100, self.n_bins + 1),
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float) + _EPSILON
        prod_counts = np.histogram(production, bins=breakpoints)[0].astype(float) + _EPSILON

        ref_dist = ref_counts / ref_counts.sum()
        prod_dist = prod_counts / prod_counts.sum()

        kl_values = kl_div(prod_dist, ref_dist)
        return float(np.sum(kl_values))

    def _compute_js_distance(
        self, reference: np.ndarray, production: np.ndarray
    ) -> float:
        """Compute Jensen-Shannon distance (symmetric KL variant).

        Uses scipy.spatial.distance.jensenshannon which returns the
        square root of the Jensen-Shannon divergence.
        """
        breakpoints = np.percentile(
            reference,
            np.linspace(0, 100, self.n_bins + 1),
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float) + _EPSILON
        prod_counts = np.histogram(production, bins=breakpoints)[0].astype(float) + _EPSILON

        ref_dist = ref_counts / ref_counts.sum()
        prod_dist = prod_counts / prod_counts.sum()

        return float(jensenshannon(ref_dist, prod_dist))

    def _classify_severity(
        self, psi: float, ks_pvalue: float, js: float
    ) -> DriftSeverity:
        """Classify drift severity based on combined test results."""
        if psi > self.psi_threshold * 2 or js > self.js_threshold * 2:
            return DriftSeverity.HIGH
        if psi > self.psi_threshold or ks_pvalue < self.ks_alpha or js > self.js_threshold:
            return DriftSeverity.MODERATE
        if psi > self.psi_threshold * 0.5 or js > self.js_threshold * 0.5:
            return DriftSeverity.LOW
        return DriftSeverity.NONE
