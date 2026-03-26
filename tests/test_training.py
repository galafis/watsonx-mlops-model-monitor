"""Tests for the training pipeline, experiment tracking, and hyperparameter optimization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.training.trainer import (
    ALGORITHM_REGISTRY,
    ModelTrainer,
    TrainingResult,
)
from src.training.hyperopt import DEFAULT_PARAM_GRIDS, HyperparameterOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Binary classification dataset (200 samples, 4 features)."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture()
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """Three-class classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(300, 4)
    y = np.digitize(X[:, 0], bins=[-0.5, 0.5])
    return X, y


# ---------------------------------------------------------------------------
# ModelTrainer — basic training
# ---------------------------------------------------------------------------

class TestModelTrainer:
    """Tests for the ModelTrainer class."""

    def test_train_random_forest(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        result = trainer.train(X, y)

        assert isinstance(result, TrainingResult)
        assert isinstance(result.model, Pipeline)
        assert result.algorithm == "random_forest"

    def test_train_logistic_regression(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="logistic_regression", random_state=42)
        result = trainer.train(X, y)

        assert result.algorithm == "logistic_regression"
        assert result.metrics["accuracy"] > 0.5

    def test_train_gradient_boosting(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="gradient_boosting", random_state=42)
        result = trainer.train(X, y)

        assert result.algorithm == "gradient_boosting"
        assert "f1" in result.metrics

    def test_invalid_algorithm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            ModelTrainer(algorithm="nonexistent_algorithm")

    def test_metrics_computed(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        result = trainer.train(X, y)

        for metric in ["accuracy", "f1", "precision", "recall"]:
            assert metric in result.metrics
            assert 0.0 <= result.metrics[metric] <= 1.0

    def test_roc_auc_computed(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        result = trainer.train(X, y)

        assert "roc_auc" in result.metrics
        assert 0.0 <= result.metrics["roc_auc"] <= 1.0

    def test_cross_validation_scores(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        result = trainer.train(X, y)

        assert isinstance(result.cv_scores, np.ndarray)
        assert len(result.cv_scores) == 5  # default cv_folds

    def test_custom_feature_names(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        names = ["age", "income", "score", "balance"]
        result = trainer.train(X, y, feature_names=names)

        assert result.feature_names == names

    def test_default_feature_names(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        result = trainer.train(X, y)

        assert result.feature_names == [f"feature_{i}" for i in range(4)]

    def test_custom_params(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(
            algorithm="random_forest",
            params={"n_estimators": 10, "max_depth": 3},
            random_state=42,
        )
        result = trainer.train(X, y)

        assert result.params == {"n_estimators": 10, "max_depth": 3}

    def test_model_can_predict(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        result = trainer.train(X, y)

        preds = result.model.predict(X[:5])
        assert len(preds) == 5
        assert all(p in [0, 1] for p in preds)


class TestModelTrainerPipeline:
    """Tests for the internal pipeline building."""

    def test_build_pipeline_has_scaler(self) -> None:
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        pipeline = trainer._build_pipeline()
        assert "scaler" in dict(pipeline.steps)
        assert "classifier" in dict(pipeline.steps)

    def test_all_algorithms_registered(self) -> None:
        expected = {"logistic_regression", "random_forest", "gradient_boosting"}
        assert set(ALGORITHM_REGISTRY.keys()) == expected


# ---------------------------------------------------------------------------
# HyperparameterOptimizer
# ---------------------------------------------------------------------------

class TestHyperparameterOptimizer:
    """Tests for GridSearchCV hyperparameter optimization."""

    def test_optimize_random_forest(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        optimizer = HyperparameterOptimizer(
            algorithm="random_forest",
            param_grid={
                "classifier__n_estimators": [10, 20],
                "classifier__max_depth": [3, 5],
            },
            cv_folds=3,
        )
        result = optimizer.optimize(X, y)

        assert isinstance(result, TrainingResult)
        assert result.algorithm == "random_forest"
        assert result.metrics["accuracy"] > 0.5

    def test_optimize_logistic_regression(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification_data
        optimizer = HyperparameterOptimizer(
            algorithm="logistic_regression",
            param_grid={
                "classifier__C": [0.1, 1.0],
                "classifier__max_iter": [200],
            },
            cv_folds=3,
        )
        result = optimizer.optimize(X, y)

        assert result.algorithm == "logistic_regression"

    def test_invalid_algorithm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            HyperparameterOptimizer(algorithm="invalid_algo")

    def test_count_combinations(self) -> None:
        optimizer = HyperparameterOptimizer(
            algorithm="random_forest",
            param_grid={
                "classifier__n_estimators": [10, 20, 50],
                "classifier__max_depth": [3, 5],
            },
        )
        assert optimizer._count_combinations() == 6

    def test_default_param_grids_exist(self) -> None:
        for algo in ALGORITHM_REGISTRY:
            assert algo in DEFAULT_PARAM_GRIDS


# ---------------------------------------------------------------------------
# ExperimentTracker (mocked MLflow)
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    """Tests for MLflow experiment tracking with mocked backend."""

    @patch("src.training.experiment.mlflow")
    def test_log_training_run(
        self,
        mock_mlflow: MagicMock,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        from src.training.experiment import ExperimentTracker

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        tracker = ExperimentTracker(
            tracking_uri="sqlite:///test.db",
            experiment_name="test-exp",
        )

        X, y = classification_data
        trainer = ModelTrainer(algorithm="random_forest", random_state=42)
        result = trainer.train(X, y)

        run_id = tracker.log_training_run(result, tags={"team": "ml"})

        assert run_id == "test-run-123"
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.sklearn.log_model.assert_called_once()

    @patch("src.training.experiment.mlflow")
    def test_get_best_run_no_experiment(self, mock_mlflow: MagicMock) -> None:
        from src.training.experiment import ExperimentTracker

        mock_mlflow.get_experiment_by_name.return_value = None

        tracker = ExperimentTracker(
            tracking_uri="sqlite:///test.db",
            experiment_name="nonexistent",
        )

        result = tracker.get_best_run()
        assert result is None
