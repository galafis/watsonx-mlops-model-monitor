"""Model training pipeline with scikit-learn classifiers and automated metric logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings

logger = structlog.get_logger(__name__)

ALGORITHM_REGISTRY: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


@dataclass
class TrainingResult:
    """Container for training pipeline outputs."""

    model: Pipeline
    metrics: dict[str, float]
    cv_scores: np.ndarray
    feature_names: list[str]
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)


class ModelTrainer:
    """Scikit-learn model training pipeline with evaluation and cross-validation.

    Supports logistic regression, random forest, and gradient boosting classifiers.
    Automatically computes accuracy, F1, precision, recall, and ROC-AUC metrics.

    Example:
        >>> trainer = ModelTrainer(algorithm="random_forest")
        >>> result = trainer.train(X, y, feature_names=["age", "income"])
        >>> print(result.metrics)
    """

    def __init__(
        self,
        algorithm: str | None = None,
        params: dict[str, Any] | None = None,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> None:
        cfg = settings.training
        self.algorithm = algorithm or cfg.get("default_algorithm", "random_forest")
        self.params = params or {}
        self.test_size = test_size or cfg.get("test_size", 0.2)
        self.random_state = random_state or cfg.get("random_state", 42)
        self.cv_folds = cfg.get("cross_validation_folds", 5)

        if self.algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )

    def _build_pipeline(self) -> Pipeline:
        """Build a scikit-learn pipeline with scaler and classifier."""
        estimator_cls = ALGORITHM_REGISTRY[self.algorithm]
        estimator_params = {**self.params}
        if "random_state" not in estimator_params and hasattr(estimator_cls(), "random_state"):
            estimator_params["random_state"] = self.random_state

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", estimator_cls(**estimator_params)),
            ]
        )

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> dict[str, float]:
        """Compute classification metrics."""
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        if y_proba is not None:
            try:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                    )
            except ValueError:
                logger.warning("roc_auc_not_computable", reason="single class in y_true")
        return metrics

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> TrainingResult:
        """Train model, evaluate on hold-out split, and run cross-validation.

        Args:
            x: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            feature_names: Optional list of feature names for documentation.

        Returns:
            TrainingResult containing the trained pipeline, metrics, and CV scores.
        """
        logger.info(
            "training_started",
            algorithm=self.algorithm,
            n_samples=x.shape[0],
            n_features=x.shape[1],
        )

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        pipeline = self._build_pipeline()
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        y_proba = (
            pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
        )

        metrics = self._compute_metrics(y_test, y_pred, y_proba)

        cv_scores = cross_val_score(
            self._build_pipeline(),
            x,
            y,
            cv=self.cv_folds,
            scoring="f1_weighted",
        )

        logger.info(
            "training_completed",
            algorithm=self.algorithm,
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            cv_mean=float(cv_scores.mean()),
        )

        return TrainingResult(
            model=pipeline,
            metrics=metrics,
            cv_scores=cv_scores,
            feature_names=feature_names or [f"feature_{i}" for i in range(x.shape[1])],
            algorithm=self.algorithm,
            params=self.params,
        )
