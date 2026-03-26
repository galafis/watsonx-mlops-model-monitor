"""MLflow experiment tracking wrapper for model training runs."""

from __future__ import annotations

from typing import Any

import mlflow
import structlog

from src.config import settings
from src.training.trainer import TrainingResult

logger = structlog.get_logger(__name__)


class ExperimentTracker:
    """Wrapper around MLflow for experiment tracking, metric logging, and model registration.

    Manages the lifecycle of MLflow runs including parameter logging,
    metric recording, and artifact storage.

    Example:
        >>> tracker = ExperimentTracker()
        >>> run_id = tracker.log_training_run(training_result, tags={"team": "ml"})
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or settings.mlflow.tracking_uri
        self.experiment_name = experiment_name or settings.mlflow.experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        logger.info(
            "experiment_tracker_initialized",
            tracking_uri=self.tracking_uri,
            experiment=self.experiment_name,
        )

    def log_training_run(
        self,
        result: TrainingResult,
        tags: dict[str, str] | None = None,
        model_name: str | None = None,
    ) -> str:
        """Log a complete training run to MLflow.

        Args:
            result: TrainingResult from the ModelTrainer.
            tags: Optional dictionary of tags to associate with the run.
            model_name: Optional name to register the model under.

        Returns:
            The MLflow run ID.
        """
        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "algorithm": result.algorithm,
                    "test_size": settings.training.get("test_size", 0.2),
                    "cv_folds": settings.training.get("cross_validation_folds", 5),
                    **{f"param_{k}": str(v) for k, v in result.params.items()},
                }
            )

            mlflow.log_metrics(result.metrics)
            mlflow.log_metric("cv_mean_f1", float(result.cv_scores.mean()))
            mlflow.log_metric("cv_std_f1", float(result.cv_scores.std()))

            if tags:
                mlflow.set_tags(tags)

            mlflow.set_tag("feature_names", ",".join(result.feature_names))
            mlflow.set_tag("n_features", str(len(result.feature_names)))

            mlflow.sklearn.log_model(
                result.model,
                artifact_path="model",
                registered_model_name=model_name,
            )

            logger.info(
                "training_run_logged",
                run_id=run.info.run_id,
                algorithm=result.algorithm,
                accuracy=result.metrics.get("accuracy"),
                model_registered=model_name is not None,
            )

            return run.info.run_id

    def get_best_run(
        self,
        metric: str = "f1",
        order: str = "DESC",
        max_results: int = 1,
    ) -> dict[str, Any] | None:
        """Retrieve the best run from the current experiment by a given metric.

        Args:
            metric: Metric name to sort by.
            order: Sort order, 'DESC' for highest first.
            max_results: Number of top runs to return.

        Returns:
            Dictionary with run info and metrics, or None if no runs exist.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=max_results,
        )

        if runs.empty:
            return None

        best = runs.iloc[0]
        return {
            "run_id": best["run_id"],
            "metrics": {
                col.replace("metrics.", ""): best[col]
                for col in runs.columns
                if col.startswith("metrics.")
            },
            "params": {
                col.replace("params.", ""): best[col]
                for col in runs.columns
                if col.startswith("params.")
            },
        }

    def get_run_metrics(self, run_id: str) -> dict[str, float]:
        """Fetch metrics for a specific MLflow run.

        Args:
            run_id: The MLflow run ID.

        Returns:
            Dictionary of metric name to value.
        """
        run = mlflow.get_run(run_id)
        return {k: float(v) for k, v in run.data.metrics.items()}
