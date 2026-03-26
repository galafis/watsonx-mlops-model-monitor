"""FastAPI model serving gateway with health checks and prediction endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from sklearn.pipeline import Pipeline

logger = structlog.get_logger(__name__)


class ModelGateway:
    """Central gateway for model inference, managing loaded models and routing.

    Wraps one or more scikit-learn pipelines and dispatches prediction requests
    through the configured routing strategy (direct, A/B, shadow).

    Example:
        >>> gateway = ModelGateway()
        >>> gateway.load_model("production", pipeline)
        >>> prediction = gateway.predict("production", [[1.0, 2.0, 3.0]])
    """

    def __init__(self) -> None:
        self._models: dict[str, Pipeline] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def load_model(
        self,
        model_id: str,
        model: Pipeline,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a model for serving.

        Args:
            model_id: Unique identifier for this model slot.
            model: Trained scikit-learn Pipeline.
            metadata: Optional metadata (version, algorithm, etc.).
        """
        self._models[model_id] = model
        self._metadata[model_id] = metadata or {}
        logger.info("model_loaded", model_id=model_id, metadata=metadata)

    def unload_model(self, model_id: str) -> None:
        """Remove a model from serving.

        Args:
            model_id: Model identifier to remove.
        """
        self._models.pop(model_id, None)
        self._metadata.pop(model_id, None)
        logger.info("model_unloaded", model_id=model_id)

    def predict(
        self,
        model_id: str,
        features: list[list[float]] | np.ndarray,
    ) -> dict[str, Any]:
        """Run inference on a loaded model.

        Args:
            model_id: Which model to use.
            features: Input feature matrix (list of samples).

        Returns:
            Dictionary with predictions, probabilities, and model metadata.

        Raises:
            KeyError: If model_id is not loaded.
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not loaded. Available: {list(self._models.keys())}")

        model = self._models[model_id]
        x = np.array(features)

        predictions = model.predict(x).tolist()
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x).tolist()

        logger.info(
            "prediction_served",
            model_id=model_id,
            n_samples=len(predictions),
        )

        return {
            "model_id": model_id,
            "predictions": predictions,
            "probabilities": probabilities,
            "metadata": self._metadata.get(model_id, {}),
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List all loaded models with metadata.

        Returns:
            List of model info dictionaries.
        """
        return [
            {
                "model_id": model_id,
                "metadata": self._metadata.get(model_id, {}),
            }
            for model_id in self._models
        ]

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded.

        Args:
            model_id: Model identifier to check.

        Returns:
            True if the model is loaded and ready for inference.
        """
        return model_id in self._models
