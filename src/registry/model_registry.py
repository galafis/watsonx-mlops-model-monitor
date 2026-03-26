"""Model versioning and registry backed by MLflow Model Registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import structlog
from mlflow.tracking import MlflowClient

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a registered model version."""

    name: str
    version: int
    run_id: str
    stage: str
    description: str
    tags: dict[str, str]
    creation_timestamp: int


class ModelRegistry:
    """Manage model registration, versioning, and stage transitions in MLflow.

    Provides methods to register models, list versions, transition stages,
    and load production models for serving.

    Example:
        >>> registry = ModelRegistry()
        >>> version = registry.register_model("credit_risk", run_id="abc123")
        >>> registry.transition_stage("credit_risk", version=1, stage="Production")
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri or settings.mlflow.tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        logger.info("model_registry_initialized", tracking_uri=self.tracking_uri)

    def register_model(
        self,
        model_name: str,
        run_id: str,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        """Register a model from an MLflow run into the Model Registry.

        Args:
            model_name: Name to register the model under.
            run_id: MLflow run ID containing the model artifact.
            description: Human-readable description of this version.
            tags: Optional tags for the model version.

        Returns:
            ModelVersion with the newly registered version metadata.
        """
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)

        if description:
            self.client.update_model_version(
                name=model_name,
                version=result.version,
                description=description,
            )

        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name, version=result.version, key=key, value=value
                )

        logger.info(
            "model_registered",
            name=model_name,
            version=result.version,
            run_id=run_id,
        )

        return ModelVersion(
            name=model_name,
            version=int(result.version),
            run_id=run_id,
            stage=result.current_stage,
            description=description,
            tags=tags or {},
            creation_timestamp=result.creation_timestamp,
        )

    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True,
    ) -> str:
        """Transition a model version to a new stage.

        Args:
            model_name: Registered model name.
            version: Version number to transition.
            stage: Target stage ('Staging', 'Production', 'Archived').
            archive_existing: If True, archive current models in the target stage.

        Returns:
            The new stage string.
        """
        result = self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )

        logger.info(
            "model_stage_transitioned",
            name=model_name,
            version=version,
            new_stage=result.current_stage,
        )

        return result.current_stage

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """List all versions of a registered model.

        Args:
            model_name: Registered model name.

        Returns:
            List of ModelVersion objects sorted by version number.
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        return [
            ModelVersion(
                name=v.name,
                version=int(v.version),
                run_id=v.run_id,
                stage=v.current_stage,
                description=v.description or "",
                tags=dict(v.tags) if v.tags else {},
                creation_timestamp=v.creation_timestamp,
            )
            for v in sorted(versions, key=lambda v: int(v.version))
        ]

    def get_production_model(self, model_name: str) -> Any:
        """Load the current production model for inference.

        Args:
            model_name: Registered model name.

        Returns:
            The loaded sklearn model/pipeline.

        Raises:
            ValueError: If no production model is found.
        """
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise ValueError(f"No production model found for '{model_name}'")

        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info(
            "production_model_loaded",
            name=model_name,
            version=versions[0].version,
        )
        return model

    def get_model_by_version(self, model_name: str, version: int) -> Any:
        """Load a specific model version.

        Args:
            model_name: Registered model name.
            version: Version number to load.

        Returns:
            The loaded sklearn model/pipeline.
        """
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)
