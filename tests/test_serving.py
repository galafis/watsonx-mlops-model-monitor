"""Tests for the model serving gateway, A/B router, and shadow mode."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.serving.ab_router import ABRouter, RoutingConfig, RoutingResult, RoutingStrategy
from src.serving.gateway import ModelGateway
from src.serving.shadow_mode import ShadowComparison, ShadowRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(random_state: int = 42) -> Pipeline:
    """Create a simple trained pipeline for testing."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=random_state)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline() -> Pipeline:
    return _make_pipeline(42)


@pytest.fixture()
def pipeline_b() -> Pipeline:
    return _make_pipeline(99)


@pytest.fixture()
def gateway(pipeline: Pipeline) -> ModelGateway:
    gw = ModelGateway()
    gw.load_model("production", pipeline, metadata={"version": 1})
    return gw


@pytest.fixture()
def gateway_with_canary(pipeline: Pipeline, pipeline_b: Pipeline) -> ModelGateway:
    gw = ModelGateway()
    gw.load_model("production", pipeline, metadata={"version": 1})
    gw.load_model("canary", pipeline_b, metadata={"version": 2})
    return gw


# ---------------------------------------------------------------------------
# ModelGateway tests
# ---------------------------------------------------------------------------


class TestModelGateway:
    """Tests for the ModelGateway class."""

    def test_load_and_predict(self, gateway: ModelGateway) -> None:
        result = gateway.predict("production", [[0.5, 0.3, -0.1]])
        assert "predictions" in result
        assert len(result["predictions"]) == 1
        assert result["model_id"] == "production"

    def test_predict_batch(self, gateway: ModelGateway) -> None:
        features = [[0.5, 0.3, -0.1], [1.0, -1.0, 0.5], [-0.5, 0.2, 0.8]]
        result = gateway.predict("production", features)
        assert len(result["predictions"]) == 3

    def test_predict_returns_probabilities(self, gateway: ModelGateway) -> None:
        result = gateway.predict("production", [[0.5, 0.3, -0.1]])
        assert result["probabilities"] is not None
        assert len(result["probabilities"]) == 1

    def test_predict_returns_metadata(self, gateway: ModelGateway) -> None:
        result = gateway.predict("production", [[0.5, 0.3, -0.1]])
        assert result["metadata"] == {"version": 1}

    def test_predict_unknown_model_raises(self, gateway: ModelGateway) -> None:
        with pytest.raises(KeyError, match="not loaded"):
            gateway.predict("nonexistent", [[0.5, 0.3, -0.1]])

    def test_load_model(self, pipeline: Pipeline) -> None:
        gw = ModelGateway()
        gw.load_model("test", pipeline, metadata={"algo": "rf"})
        assert gw.is_loaded("test")
        assert not gw.is_loaded("other")

    def test_unload_model(self, gateway: ModelGateway) -> None:
        assert gateway.is_loaded("production")
        gateway.unload_model("production")
        assert not gateway.is_loaded("production")

    def test_unload_nonexistent_no_error(self, gateway: ModelGateway) -> None:
        gateway.unload_model("nonexistent")  # should not raise

    def test_list_models(self, gateway_with_canary: ModelGateway) -> None:
        models = gateway_with_canary.list_models()
        assert len(models) == 2
        model_ids = {m["model_id"] for m in models}
        assert model_ids == {"production", "canary"}

    def test_is_loaded(self, gateway: ModelGateway) -> None:
        assert gateway.is_loaded("production") is True
        assert gateway.is_loaded("canary") is False


class TestModelGatewayNumpy:
    """Test gateway with numpy array inputs."""

    def test_predict_with_numpy(self, gateway: ModelGateway) -> None:
        features = np.array([[0.5, 0.3, -0.1]])
        result = gateway.predict("production", features)
        assert len(result["predictions"]) == 1


# ---------------------------------------------------------------------------
# ABRouter tests
# ---------------------------------------------------------------------------


class TestABRouterDirect:
    """Tests for direct routing strategy."""

    def test_direct_routing(self, gateway: ModelGateway) -> None:
        config = RoutingConfig(strategy=RoutingStrategy.DIRECT)
        router = ABRouter(gateway, config=config)
        result = router.route([[0.5, 0.3, -0.1]])

        assert isinstance(result, RoutingResult)
        assert result.routed_to == "production"
        assert result.strategy == "direct"

    def test_stats_tracked(self, gateway: ModelGateway) -> None:
        config = RoutingConfig(strategy=RoutingStrategy.DIRECT)
        router = ABRouter(gateway, config=config)

        for _ in range(5):
            router.route([[0.5, 0.3, -0.1]])

        stats = router.get_stats()
        assert stats["total_requests"] == 5


class TestABRouterCanary:
    """Tests for canary routing strategy."""

    def test_canary_routing_uses_production_by_default(
        self, gateway_with_canary: ModelGateway
    ) -> None:
        config = RoutingConfig(
            strategy=RoutingStrategy.CANARY,
            canary_weight=0.0,
        )
        router = ABRouter(gateway_with_canary, config=config)
        result = router.route([[0.5, 0.3, -0.1]])

        assert result.routed_to == "production"
        assert result.strategy == "canary"

    def test_canary_routing_can_use_canary(self, gateway_with_canary: ModelGateway) -> None:
        config = RoutingConfig(
            strategy=RoutingStrategy.CANARY,
            canary_weight=1.0,
        )
        router = ABRouter(gateway_with_canary, config=config)
        result = router.route([[0.5, 0.3, -0.1]])

        assert result.routed_to == "canary"

    def test_canary_falls_back_when_not_loaded(self, gateway: ModelGateway) -> None:
        config = RoutingConfig(
            strategy=RoutingStrategy.CANARY,
            canary_weight=1.0,
        )
        router = ABRouter(gateway, config=config)
        result = router.route([[0.5, 0.3, -0.1]])

        assert result.routed_to == "production"

    def test_update_canary_weight(self, gateway_with_canary: ModelGateway) -> None:
        config = RoutingConfig(strategy=RoutingStrategy.CANARY, canary_weight=0.1)
        router = ABRouter(gateway_with_canary, config=config)

        router.update_canary_weight(0.5)
        assert router.config.canary_weight == 0.5

    def test_canary_weight_clamped(self, gateway_with_canary: ModelGateway) -> None:
        config = RoutingConfig(strategy=RoutingStrategy.CANARY)
        router = ABRouter(gateway_with_canary, config=config)

        router.update_canary_weight(1.5)
        assert router.config.canary_weight == 1.0

        router.update_canary_weight(-0.5)
        assert router.config.canary_weight == 0.0


class TestABRouterBlueGreen:
    """Tests for blue-green routing strategy."""

    def test_blue_green_routing(self, gateway_with_canary: ModelGateway) -> None:
        config = RoutingConfig(strategy=RoutingStrategy.BLUE_GREEN)
        router = ABRouter(gateway_with_canary, config=config)
        result = router.route([[0.5, 0.3, -0.1]])

        assert result.routed_to == "production"
        assert result.strategy == "blue_green"

    def test_switch_blue_green(self, gateway_with_canary: ModelGateway) -> None:
        config = RoutingConfig(strategy=RoutingStrategy.BLUE_GREEN)
        router = ABRouter(gateway_with_canary, config=config)

        new_primary = router.switch_blue_green()
        assert new_primary == "canary"
        assert router.config.primary_model == "canary"
        assert router.config.candidate_model == "production"


class TestABRouterShadow:
    """Tests for shadow routing strategy."""

    def test_shadow_routing(self, gateway_with_canary: ModelGateway) -> None:
        config = RoutingConfig(strategy=RoutingStrategy.SHADOW)
        router = ABRouter(gateway_with_canary, config=config)
        result = router.route([[0.5, 0.3, -0.1]])

        assert result.routed_to == "production"
        assert result.strategy == "shadow"
        assert result.shadow_prediction is not None

    def test_shadow_with_no_candidate(self, gateway: ModelGateway) -> None:
        config = RoutingConfig(strategy=RoutingStrategy.SHADOW)
        router = ABRouter(gateway, config=config)
        result = router.route([[0.5, 0.3, -0.1]])

        assert result.routed_to == "production"
        assert result.shadow_prediction is None


# ---------------------------------------------------------------------------
# ShadowRunner tests
# ---------------------------------------------------------------------------


class TestShadowRunner:
    """Tests for the shadow deployment runner."""

    def test_run_shadow_basic(self, gateway_with_canary: ModelGateway) -> None:
        runner = ShadowRunner(gateway_with_canary)
        result = runner.run_shadow("canary", [[0.5, 0.3, -0.1]])

        assert result is not None
        assert "predictions" in result

    def test_run_shadow_not_loaded(self, gateway: ModelGateway) -> None:
        runner = ShadowRunner(gateway)
        result = runner.run_shadow("nonexistent", [[0.5, 0.3, -0.1]])

        assert result is None

    def test_shadow_comparison_recording(self, gateway_with_canary: ModelGateway) -> None:
        runner = ShadowRunner(gateway_with_canary)

        prod_preds = [1, 0, 1]
        runner.run_shadow(
            "canary",
            [[0.5, 0.3, -0.1], [1.0, -1.0, 0.5], [-0.5, 0.2, 0.8]],
            production_predictions=prod_preds,
        )

        comp = runner.get_comparison()
        assert comp.total_requests == 3
        assert comp.agreement_count + comp.divergence_count == 3

    def test_shadow_comparison_reset(self, gateway_with_canary: ModelGateway) -> None:
        runner = ShadowRunner(gateway_with_canary)
        runner.run_shadow(
            "canary",
            [[0.5, 0.3, -0.1]],
            production_predictions=[1],
        )

        runner.reset_comparison()
        comp = runner.get_comparison()
        assert comp.total_requests == 0

    def test_shadow_summary(self, gateway_with_canary: ModelGateway) -> None:
        runner = ShadowRunner(gateway_with_canary)
        summary = runner.get_summary()

        assert "total_requests" in summary
        assert "agreement_rate" in summary
        assert "divergence_rate" in summary


class TestShadowComparison:
    """Tests for the ShadowComparison dataclass."""

    def test_default_rates(self) -> None:
        comp = ShadowComparison()
        assert comp.agreement_rate == 0.0
        assert comp.divergence_rate == 0.0

    def test_rates_with_data(self) -> None:
        comp = ShadowComparison(
            total_requests=10,
            agreement_count=7,
            divergence_count=3,
        )
        assert comp.agreement_rate == pytest.approx(0.7)
        assert comp.divergence_rate == pytest.approx(0.3)
