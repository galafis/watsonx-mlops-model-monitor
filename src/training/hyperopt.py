"""Hyperparameter optimization via GridSearchCV with MLflow logging."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from sklearn.model_selection import GridSearchCV

from src.config import settings
from src.training.trainer import ALGORITHM_REGISTRY, ModelTrainer, TrainingResult

logger = structlog.get_logger(__name__)

DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__max_iter": [200, 500],
    },
    "random_forest": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
    },
}


class HyperparameterOptimizer:
    """Grid search hyperparameter optimization with cross-validation.

    Uses scikit-learn's GridSearchCV to find optimal parameters for a given
    algorithm, then retrains the full model with those parameters.

    Example:
        >>> optimizer = HyperparameterOptimizer(algorithm="random_forest")
        >>> result = optimizer.optimize(X, y, feature_names=["age", "income"])
        >>> print(result.params)
    """

    def __init__(
        self,
        algorithm: str | None = None,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int | None = None,
        scoring: str = "f1_weighted",
    ) -> None:
        cfg = settings.training
        self.algorithm = algorithm or cfg.get("default_algorithm", "random_forest")
        self.cv_folds = cv_folds or cfg.get("cross_validation_folds", 5)
        self.scoring = scoring

        if self.algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )

        self.param_grid = param_grid or DEFAULT_PARAM_GRIDS.get(self.algorithm, {})

    def optimize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> TrainingResult:
        """Run grid search and return a TrainingResult with the best model.

        Args:
            x: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            feature_names: Optional list of feature names.

        Returns:
            TrainingResult with the best hyperparameters applied.
        """
        logger.info(
            "hyperopt_started",
            algorithm=self.algorithm,
            n_combinations=self._count_combinations(),
            cv_folds=self.cv_folds,
        )

        base_trainer = ModelTrainer(algorithm=self.algorithm)
        pipeline = base_trainer._build_pipeline()

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            refit=False,
            return_train_score=True,
        )

        grid_search.fit(x, y)

        best_params = grid_search.best_params_
        clean_params = {
            k.replace("classifier__", ""): v for k, v in best_params.items()
        }

        logger.info(
            "hyperopt_completed",
            best_score=grid_search.best_score_,
            best_params=clean_params,
        )

        trainer = ModelTrainer(algorithm=self.algorithm, params=clean_params)
        result = trainer.train(x, y, feature_names=feature_names)

        return result

    def _count_combinations(self) -> int:
        """Count total number of hyperparameter combinations."""
        if not self.param_grid:
            return 0
        count = 1
        for values in self.param_grid.values():
            count *= len(values)
        return count
