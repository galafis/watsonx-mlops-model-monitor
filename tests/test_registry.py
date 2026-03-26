"""Tests for the model registry and versioning module with mocked MLflow backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.registry.model_registry import ModelRegistry, ModelVersion
from src.registry.versioning import VersionComparison, VersionManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mlflow_client() -> MagicMock:
    """Create a mocked MlflowClient."""
    client = MagicMock()
    return client


# ---------------------------------------------------------------------------
# ModelRegistry tests (all MLflow calls mocked)
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Tests for ModelRegistry with mocked MLflow."""

    @patch("src.registry.model_registry.mlflow")
    @patch("src.registry.model_registry.MlflowClient")
    def test_register_model(self, mock_client_cls: MagicMock, mock_mlflow: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.version = "1"
        mock_result.current_stage = "None"
        mock_result.creation_timestamp = 1700000000
        mock_mlflow.register_model.return_value = mock_result

        registry = ModelRegistry(tracking_uri="sqlite:///test.db")
        version = registry.register_model(
            model_name="credit_risk",
            run_id="run-abc-123",
            description="Initial model",
            tags={"team": "ml"},
        )

        assert isinstance(version, ModelVersion)
        assert version.name == "credit_risk"
        assert version.version == 1
        assert version.run_id == "run-abc-123"
        assert version.description == "Initial model"
        assert version.tags == {"team": "ml"}

        mock_mlflow.register_model.assert_called_once_with("runs:/run-abc-123/model", "credit_risk")

    @patch("src.registry.model_registry.mlflow")
    @patch("src.registry.model_registry.MlflowClient")
    def test_register_model_no_description(
        self, mock_client_cls: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.version = "2"
        mock_result.current_stage = "None"
        mock_result.creation_timestamp = 1700000001
        mock_mlflow.register_model.return_value = mock_result

        registry = ModelRegistry(tracking_uri="sqlite:///test.db")
        version = registry.register_model("model_a", "run-xyz")

        assert version.version == 2
        assert version.description == ""
        # update_model_version should NOT be called without description
        registry.client.update_model_version.assert_not_called()

    @patch("src.registry.model_registry.mlflow")
    @patch("src.registry.model_registry.MlflowClient")
    def test_transition_stage(self, mock_client_cls: MagicMock, mock_mlflow: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.current_stage = "Production"

        registry = ModelRegistry(tracking_uri="sqlite:///test.db")
        registry.client.transition_model_version_stage.return_value = mock_response

        stage = registry.transition_stage("credit_risk", version=1, stage="Production")

        assert stage == "Production"
        registry.client.transition_model_version_stage.assert_called_once_with(
            name="credit_risk",
            version="1",
            stage="Production",
            archive_existing_versions=True,
        )

    @patch("src.registry.model_registry.mlflow")
    @patch("src.registry.model_registry.MlflowClient")
    def test_list_versions(self, mock_client_cls: MagicMock, mock_mlflow: MagicMock) -> None:
        v1 = MagicMock()
        v1.name = "model"
        v1.version = "1"
        v1.run_id = "run1"
        v1.current_stage = "Archived"
        v1.description = "v1"
        v1.tags = {}
        v1.creation_timestamp = 1000

        v2 = MagicMock()
        v2.name = "model"
        v2.version = "2"
        v2.run_id = "run2"
        v2.current_stage = "Production"
        v2.description = "v2"
        v2.tags = {"team": "ml"}
        v2.creation_timestamp = 2000

        registry = ModelRegistry(tracking_uri="sqlite:///test.db")
        registry.client.search_model_versions.return_value = [v2, v1]

        versions = registry.list_versions("model")

        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2
        assert versions[1].stage == "Production"

    @patch("src.registry.model_registry.mlflow")
    @patch("src.registry.model_registry.MlflowClient")
    def test_get_production_model(self, mock_client_cls: MagicMock, mock_mlflow: MagicMock) -> None:
        mock_version = MagicMock()
        mock_version.version = "3"

        registry = ModelRegistry(tracking_uri="sqlite:///test.db")
        registry.client.get_latest_versions.return_value = [mock_version]

        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model

        model = registry.get_production_model("credit_risk")

        assert model == mock_model
        mock_mlflow.sklearn.load_model.assert_called_once_with("models:/credit_risk/Production")

    @patch("src.registry.model_registry.mlflow")
    @patch("src.registry.model_registry.MlflowClient")
    def test_get_production_model_not_found(
        self, mock_client_cls: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        registry = ModelRegistry(tracking_uri="sqlite:///test.db")
        registry.client.get_latest_versions.return_value = []

        with pytest.raises(ValueError, match="No production model found"):
            registry.get_production_model("credit_risk")

    @patch("src.registry.model_registry.mlflow")
    @patch("src.registry.model_registry.MlflowClient")
    def test_get_model_by_version(self, mock_client_cls: MagicMock, mock_mlflow: MagicMock) -> None:
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model

        registry = ModelRegistry(tracking_uri="sqlite:///test.db")
        model = registry.get_model_by_version("credit_risk", version=2)

        assert model == mock_model
        mock_mlflow.sklearn.load_model.assert_called_once_with("models:/credit_risk/2")


# ---------------------------------------------------------------------------
# VersionManager tests
# ---------------------------------------------------------------------------


class TestVersionManager:
    """Tests for model version comparison and promotion logic."""

    @patch("src.registry.versioning.ModelRegistry")
    def test_promote_candidate(self, mock_registry_cls: MagicMock) -> None:
        mock_registry = MagicMock()
        mock_registry.transition_stage.return_value = "Production"

        manager = VersionManager(registry=mock_registry)
        stage = manager.promote_candidate("credit_risk", version=3)

        assert stage == "Production"
        mock_registry.transition_stage.assert_called_once_with(
            model_name="credit_risk",
            version=3,
            stage="Production",
            archive_existing=True,
        )

    @patch("src.registry.versioning.ModelRegistry")
    def test_rollback(self, mock_registry_cls: MagicMock) -> None:
        mock_registry = MagicMock()
        mock_registry.transition_stage.return_value = "Production"

        manager = VersionManager(registry=mock_registry)
        stage = manager.rollback("credit_risk", to_version=1)

        assert stage == "Production"
        mock_registry.transition_stage.assert_called_once_with(
            model_name="credit_risk",
            version=1,
            stage="Production",
            archive_existing=True,
        )

    @patch("src.registry.versioning.ExperimentTracker")
    @patch("src.registry.versioning.ModelRegistry")
    def test_compare_versions_should_promote(
        self, mock_registry_cls: MagicMock, mock_tracker_cls: MagicMock
    ) -> None:
        mock_registry = MagicMock()

        v2 = MagicMock()
        v2.version = 2
        v2.run_id = "run-2"

        v3 = MagicMock()
        v3.version = 3
        v3.run_id = "run-3"

        mock_registry.list_versions.return_value = [v2, v3]

        mock_tracker = MagicMock()
        mock_tracker.get_run_metrics.side_effect = [
            {"f1": 0.90, "accuracy": 0.92},  # candidate (v3)
            {"f1": 0.85, "accuracy": 0.88},  # production (v2)
        ]
        mock_tracker_cls.return_value = mock_tracker

        manager = VersionManager(registry=mock_registry)
        comparison = manager.compare_versions(
            "credit_risk",
            candidate_version=3,
            production_version=2,
            primary_metric="f1",
            min_improvement=0.01,
        )

        assert isinstance(comparison, VersionComparison)
        assert comparison.should_promote is True
        assert comparison.candidate_version == 3
        assert comparison.production_version == 2

    @patch("src.registry.versioning.ExperimentTracker")
    @patch("src.registry.versioning.ModelRegistry")
    def test_compare_versions_should_not_promote(
        self, mock_registry_cls: MagicMock, mock_tracker_cls: MagicMock
    ) -> None:
        mock_registry = MagicMock()

        v2 = MagicMock()
        v2.version = 2
        v2.run_id = "run-2"

        v3 = MagicMock()
        v3.version = 3
        v3.run_id = "run-3"

        mock_registry.list_versions.return_value = [v2, v3]

        mock_tracker = MagicMock()
        mock_tracker.get_run_metrics.side_effect = [
            {"f1": 0.84, "accuracy": 0.86},  # candidate (v3)
            {"f1": 0.85, "accuracy": 0.88},  # production (v2)
        ]
        mock_tracker_cls.return_value = mock_tracker

        manager = VersionManager(registry=mock_registry)
        comparison = manager.compare_versions(
            "credit_risk",
            candidate_version=3,
            production_version=2,
        )

        assert comparison.should_promote is False

    @patch("src.registry.versioning.ModelRegistry")
    def test_compare_versions_not_found(self, mock_registry_cls: MagicMock) -> None:
        mock_registry = MagicMock()
        mock_registry.list_versions.return_value = []

        manager = VersionManager(registry=mock_registry)

        with pytest.raises(ValueError, match="not found"):
            manager.compare_versions("credit_risk", candidate_version=99, production_version=1)


# ---------------------------------------------------------------------------
# ModelVersion dataclass
# ---------------------------------------------------------------------------


class TestModelVersionDataclass:
    """Tests for the ModelVersion dataclass."""

    def test_model_version_creation(self) -> None:
        mv = ModelVersion(
            name="test_model",
            version=1,
            run_id="run-123",
            stage="None",
            description="First version",
            tags={"team": "ml"},
            creation_timestamp=1700000000,
        )
        assert mv.name == "test_model"
        assert mv.version == 1
        assert mv.run_id == "run-123"
        assert mv.stage == "None"
        assert mv.tags == {"team": "ml"}
