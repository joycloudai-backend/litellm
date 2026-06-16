"""
Tests for BytePlus video billing integration.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from litellm.proxy.spend_tracking.volcengine_video_billing import (
    ARK_VIDEO_PROVIDERS,
    BYTEPLUS_VIDEO_DEFAULT_PRICING_MODEL,
    VOLCENGINE_VIDEO_DEFAULT_PRICING_MODEL,
    VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS,
    VolcengineVideoBillingManager,
    _candidate_pricing_models,
    _is_1080p_resolution,
    _normalize_pricing_model,
)
from litellm.types.videos.main import VideoObject


class TestBytePlusBillingConstants:
    """Test BytePlus billing constants are correctly defined."""

    def test_byteplus_default_pricing_model(self):
        assert BYTEPLUS_VIDEO_DEFAULT_PRICING_MODEL == "byteplus/dreamina-seedance-2.0"

    def test_ark_video_providers_includes_byteplus(self):
        assert "byteplus" in ARK_VIDEO_PROVIDERS
        assert "volcengine" in ARK_VIDEO_PROVIDERS

    def test_runtime_pricing_models_has_byteplus(self):
        assert (
            "byteplus/dreamina-seedance-2.0" in VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS
        )
        assert (
            "byteplus/dreamina-seedance-2.0-fast"
            in VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS
        )

    def test_byteplus_runtime_pricing_is_usd(self):
        model = VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS[
            "byteplus/dreamina-seedance-2.0"
        ]
        assert model["provider_pricing_currency"] == "USD"
        assert model["litellm_provider"] == "byteplus"

    def test_byteplus_fast_runtime_pricing_is_usd(self):
        model = VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS[
            "byteplus/dreamina-seedance-2.0-fast"
        ]
        assert model["provider_pricing_currency"] == "USD"
        assert (
            model["volcengine_video_output_cost_per_million_tokens_without_input_video"]
            == 5.6
        )
        assert (
            model["volcengine_video_output_cost_per_million_tokens_with_input_video"]
            == 3.3
        )


class TestBytePlusPricingModelNormalization:
    """Test model name normalization for BytePlus."""

    def test_normalize_byteplus_model_preserves_prefix(self):
        assert (
            _normalize_pricing_model("byteplus/dreamina-seedance-2.0")
            == "byteplus/dreamina-seedance-2.0"
        )

    def test_normalize_volcengine_model_preserves_prefix(self):
        assert (
            _normalize_pricing_model("volcengine/doubao-seedance-2.0")
            == "volcengine/doubao-seedance-2.0"
        )

    def test_normalize_bare_model_adds_volcengine_prefix(self):
        assert (
            _normalize_pricing_model("doubao-seedance-2.0")
            == "volcengine/doubao-seedance-2.0"
        )

    def test_candidate_pricing_models_includes_byteplus(self):
        candidates = _candidate_pricing_models("byteplus/dreamina-seedance-2.0")
        assert "byteplus/dreamina-seedance-2.0" in candidates

    def test_candidate_pricing_models_dotted_variant(self):
        candidates = _candidate_pricing_models("byteplus/dreamina-seedance-2-0-260128")
        assert "byteplus/dreamina-seedance-2-0-260128" in candidates
        assert "byteplus/dreamina-seedance-2.0-260128" in candidates


class TestBytePlusShouldHandleSuccessEvent:
    """Test that the billing manager handles BytePlus video events."""

    def _make_manager(self):
        return VolcengineVideoBillingManager(
            prisma_client=MagicMock(),
            llm_router=MagicMock(),
            db_spend_update_writer=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )

    def test_should_handle_byteplus_video_create(self):
        manager = self._make_manager()
        kwargs = {
            "call_type": "avideo_generation",
            "custom_llm_provider": "byteplus",
        }
        assert manager.should_handle_success_event(kwargs) is True

    def test_should_handle_byteplus_video_status(self):
        manager = self._make_manager()
        kwargs = {
            "call_type": "avideo_status",
            "custom_llm_provider": "byteplus",
        }
        assert manager.should_handle_success_event(kwargs) is True

    def test_should_handle_volcengine_still_works(self):
        manager = self._make_manager()
        kwargs = {
            "call_type": "avideo_generation",
            "custom_llm_provider": "volcengine",
        }
        assert manager.should_handle_success_event(kwargs) is True

    def test_should_not_handle_other_providers(self):
        manager = self._make_manager()
        kwargs = {
            "call_type": "avideo_generation",
            "custom_llm_provider": "openai",
        }
        assert manager.should_handle_success_event(kwargs) is False

    def test_should_not_handle_non_video_call_types(self):
        manager = self._make_manager()
        kwargs = {
            "call_type": "acompletion",
            "custom_llm_provider": "byteplus",
        }
        assert manager.should_handle_success_event(kwargs) is False

    def test_should_handle_byteplus_from_litellm_params(self):
        manager = self._make_manager()
        kwargs = {
            "call_type": "avideo_generation",
            "litellm_params": {"custom_llm_provider": "byteplus"},
        }
        assert manager.should_handle_success_event(kwargs) is True


class TestBytePlusCredentialLookup:
    """Test that BytePlus credentials are correctly resolved."""

    def _make_manager(self):
        mock_router = MagicMock()
        mock_router.get_deployment_credentials_with_provider.return_value = {
            "api_key": "test-byteplus-key",
            "custom_llm_provider": "byteplus",
            "api_base": "https://ark.ap-southeast.bytepluses.com/api/v3",
        }
        return VolcengineVideoBillingManager(
            prisma_client=MagicMock(),
            llm_router=mock_router,
            db_spend_update_writer=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )

    def test_get_credentials_for_byteplus_model(self):
        manager = self._make_manager()
        task = Mock()
        task.model_id = "model-123"
        task.model_group = "seedance-2.0"
        task.model = "seedance-2.0"
        task.provider_model = "dreamina-seedance-2-0-260128"

        credentials = manager._get_volcengine_credentials(task)
        assert credentials is not None
        assert credentials["custom_llm_provider"] == "byteplus"
        assert credentials["api_key"] == "test-byteplus-key"


class TestResolutionDetection:
    """Test that 1080p outputs are detected across the formats the API can return."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1080p", True),
            ("1080", True),
            ("1920x1080", True),
            ("1080x1920", True),
            ("720p", False),
            ("480p", False),
            ("1280x720", False),
            ("3840x2160", False),
            (None, False),
            ("", False),
            ("garbage", False),
        ],
    )
    def test_is_1080p_resolution(self, value, expected):
        assert _is_1080p_resolution(value) is expected


class TestResolutionAwarePricing:
    """
    byteplus/dreamina-seedance-2.0 charges a 1080p premium while 480p/720p share a
    base rate; the -fast variant has no resolution tiers.
    """

    def _make_manager(self):
        return VolcengineVideoBillingManager(
            prisma_client=MagicMock(),
            llm_router=MagicMock(),
            db_spend_update_writer=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )

    @pytest.mark.parametrize(
        "has_input_video, resolution, expected",
        [
            (False, "720p", 7.0),
            (False, "480p", 7.0),
            (False, None, 7.0),
            (True, "720p", 4.3),
            (True, "480p", 4.3),
            (False, "1080p", 7.7),
            (False, "1920x1080", 7.7),
            (True, "1080p", 4.7),
            (True, "1080x1920", 4.7),
        ],
    )
    def test_seedance_2_0_pricing_by_resolution(
        self, has_input_video, resolution, expected
    ):
        manager = self._make_manager()
        unit_price, currency = manager._resolve_pricing_snapshot(
            pricing_model="byteplus/dreamina-seedance-2.0",
            has_input_video=has_input_video,
            resolution=resolution,
        )
        assert unit_price == expected
        assert currency == "USD"

    @pytest.mark.parametrize(
        "has_input_video, resolution, expected",
        [
            (False, "720p", 5.6),
            (False, "1080p", 5.6),
            (True, "720p", 3.3),
            (True, "1080p", 3.3),
        ],
    )
    def test_seedance_2_0_fast_ignores_resolution(
        self, has_input_video, resolution, expected
    ):
        manager = self._make_manager()
        unit_price, _ = manager._resolve_pricing_snapshot(
            pricing_model="byteplus/dreamina-seedance-2.0-fast",
            has_input_video=has_input_video,
            resolution=resolution,
        )
        assert unit_price == expected

    @pytest.mark.parametrize(
        "has_input_video, resolution, expected",
        [
            (False, "720p", 46.0),
            (False, "480p", 46.0),
            (False, None, 46.0),
            (True, "720p", 28.0),
            (False, "1080p", 51.0),
            (False, "1920x1080", 51.0),
            (True, "1080p", 31.0),
        ],
    )
    def test_volcengine_doubao_seedance_2_0_pricing_by_resolution(
        self, has_input_video, resolution, expected
    ):
        manager = self._make_manager()
        unit_price, currency = manager._resolve_pricing_snapshot(
            pricing_model="volcengine/doubao-seedance-2.0",
            has_input_video=has_input_video,
            resolution=resolution,
        )
        assert unit_price == expected
        assert currency == "CNY"

    @pytest.mark.parametrize(
        "has_input_video, resolution, expected",
        [
            (False, "720p", 37.0),
            (False, "1080p", 37.0),
            (True, "720p", 22.0),
            (True, "1080p", 22.0),
        ],
    )
    def test_volcengine_doubao_seedance_2_0_fast_ignores_resolution(
        self, has_input_video, resolution, expected
    ):
        manager = self._make_manager()
        unit_price, _ = manager._resolve_pricing_snapshot(
            pricing_model="volcengine/doubao-seedance-2.0-fast",
            has_input_video=has_input_video,
            resolution=resolution,
        )
        assert unit_price == expected


class TestDeploymentModelDoesNotShadowPricing:
    """
    The proxy registers each deployment's litellm_params.model into
    litellm.model_cost without the Volcengine video pricing keys. That generic
    entry must not shadow the dotted runtime pricing entry that actually carries
    the per-million-token rates.
    """

    def _make_manager(self):
        return VolcengineVideoBillingManager(
            prisma_client=MagicMock(),
            llm_router=MagicMock(),
            db_spend_update_writer=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )

    def test_versioned_deployment_model_falls_through_to_runtime_pricing(self):
        import litellm

        manager = self._make_manager()
        deployment_model = "byteplus/dreamina-seedance-2-0-260128"
        litellm.register_model(
            model_cost={
                deployment_model: {
                    "litellm_provider": "byteplus",
                    "mode": "video_generation",
                }
            }
        )
        try:
            unit_price, currency = manager._resolve_pricing_snapshot(
                pricing_model=deployment_model,
                has_input_video=False,
                resolution="720p",
            )
        finally:
            litellm.model_cost.pop(deployment_model, None)

        assert unit_price == 7.0
        assert currency == "USD"


class TestFinalUnitPriceResolution:
    """The completed task's reported resolution drives the billed price."""

    def _make_manager(self):
        return VolcengineVideoBillingManager(
            prisma_client=MagicMock(),
            llm_router=MagicMock(),
            db_spend_update_writer=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )

    def _make_task(self, **overrides):
        task = Mock()
        task.video_id = "vid-1"
        task.pricing_model = "byteplus/dreamina-seedance-2.0"
        task.has_input_video = False
        task.price_per_million_tokens = 7.0
        for key, value in overrides.items():
            setattr(task, key, value)
        return task

    def _make_video_response(self, resolution=None, usage=None):
        video = VideoObject(id="vid-1", object="video", status="completed", usage=usage)
        if resolution is not None:
            video._hidden_params = {"resolution": resolution}
        return video

    def test_provider_1080p_resolution_overrides_snapshot(self):
        manager = self._make_manager()
        task = self._make_task(price_per_million_tokens=7.0)
        video = self._make_video_response(resolution="1080p")
        assert manager._resolve_final_unit_price(task=task, video_response=video) == 7.7

    def test_provider_720p_resolution_uses_base_rate(self):
        manager = self._make_manager()
        task = self._make_task(has_input_video=True, price_per_million_tokens=4.7)
        video = self._make_video_response(resolution="720p")
        assert manager._resolve_final_unit_price(task=task, video_response=video) == 4.3

    def test_resolution_from_usage_object(self):
        manager = self._make_manager()
        task = self._make_task()
        video = self._make_video_response(usage={"video_resolution": "1080p"})
        assert manager._resolve_final_unit_price(task=task, video_response=video) == 7.7

    def test_missing_resolution_falls_back_to_snapshot(self):
        manager = self._make_manager()
        task = self._make_task(price_per_million_tokens=7.7)
        video = self._make_video_response(resolution=None)
        assert manager._resolve_final_unit_price(task=task, video_response=video) == 7.7


class TestBytePlusUsdConversion:
    """Test that USD pricing is handled correctly."""

    def test_usd_passthrough(self):
        from litellm.proxy.spend_tracking.volcengine_video_billing import (
            _convert_provider_spend_to_usd,
        )

        # USD should pass through unchanged
        assert _convert_provider_spend_to_usd(10.0, "USD") == 10.0

    def test_cny_conversion_still_works(self):
        from litellm.proxy.spend_tracking.volcengine_video_billing import (
            _convert_provider_spend_to_usd,
        )

        result = _convert_provider_spend_to_usd(72.0, "CNY")
        assert result == pytest.approx(10.0, abs=0.1)
