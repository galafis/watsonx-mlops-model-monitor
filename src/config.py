"""Application configuration loaded from environment variables and settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML configuration from config/settings.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


_yaml = _load_yaml_config()


class WatsonxSettings(BaseSettings):
    """IBM Watsonx AI credentials and model configuration."""

    api_key: str = Field(default="", alias="WATSONX_API_KEY")
    project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        alias="WATSONX_URL",
    )
    generation_model: str = Field(default="ibm/granite-13b-chat-v2")
    embedding_model: str = Field(default="ibm/slate-125m-english-rtrvr")


class MLflowSettings(BaseSettings):
    """MLflow tracking server configuration."""

    tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(
        default="watsonx-model-monitor", alias="MLFLOW_EXPERIMENT_NAME"
    )


class AppSettings(BaseSettings):
    """General application settings."""

    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8080, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")


class Settings:
    """Aggregated application settings from env vars and YAML config."""

    def __init__(self) -> None:
        self.watsonx = WatsonxSettings()
        self.mlflow = MLflowSettings()
        self.app = AppSettings()
        self.yaml = _yaml

    @property
    def training(self) -> dict[str, Any]:
        """Training pipeline configuration."""
        return self.yaml.get("training", {})

    @property
    def monitoring(self) -> dict[str, Any]:
        """Monitoring thresholds and configuration."""
        return self.yaml.get("monitoring", {})

    @property
    def drift_config(self) -> dict[str, Any]:
        """Drift detection thresholds."""
        return self.monitoring.get("drift", {})

    @property
    def fairness_config(self) -> dict[str, Any]:
        """Fairness monitoring configuration."""
        return self.monitoring.get("fairness", {})

    @property
    def performance_config(self) -> dict[str, Any]:
        """Performance tracking configuration."""
        return self.monitoring.get("performance", {})

    @property
    def deployment(self) -> dict[str, Any]:
        """Deployment strategy configuration."""
        return self.yaml.get("deployment", {})

    @property
    def governance(self) -> dict[str, Any]:
        """Governance and lifecycle configuration."""
        return self.yaml.get("governance", {})

    @property
    def alerting(self) -> dict[str, Any]:
        """Alert engine configuration."""
        return self.yaml.get("alerting", {})

    @property
    def generation_params(self) -> dict[str, Any]:
        """Watsonx generation model parameters."""
        return self.yaml.get("watsonx", {}).get("generation", {}).get("parameters", {})


settings = Settings()
