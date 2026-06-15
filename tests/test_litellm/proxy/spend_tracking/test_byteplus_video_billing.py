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
    _normalize_pricing_model,
)


class TestBytePlusBillingConstants:
    """Test BytePlus billing constants are correctly defined."""

    def test_byteplus_default_pricing_model(self):
        assert BYTEPLUS_VIDEO_DEFAULT_PRICING_MODEL == "byteplus/dreamina-seedance-2.0"

    def test_ark_video_providers_includes_byteplus(self):
        assert "byteplus" in ARK_VIDEO_PROVIDERS
        assert "volcengine" in ARK_VIDEO_PROVIDERS

    def test_runtime_pricing_models_has_byteplus(self):
        assert "byteplus/dreamina-seedance-2.0" in VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS
        assert "byteplus/dreamina-seedance-2.0-fast" in VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS

    def test_byteplus_runtime_pricing_is_usd(self):
        model = VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS["byteplus/dreamina-seedance-2.0"]
        assert model["provider_pricing_currency"] == "USD"
        assert model["litellm_provider"] == "byteplus"

    def test_byteplus_fast_runtime_pricing_is_usd(self):
        model = VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS["byteplus/dreamina-seedance-2.0-fast"]
        assert model["provider_pricing_currency"] == "USD"
        assert model["volcengine_video_output_cost_per_million_tokens_without_input_video"] == 5.1
        assert model["volcengine_video_output_cost_per_million_tokens_with_input_video"] == 3.1


class TestBytePlusPricingModelNormalization:
    """Test model name normalization for BytePlus."""

    def test_normalize_byteplus_model_preserves_prefix(self):
        assert _normalize_pricing_model("byteplus/dreamina-seedance-2.0") == "byteplus/dreamina-seedance-2.0"

    def test_normalize_volcengine_model_preserves_prefix(self):
        assert _normalize_pricing_model("volcengine/doubao-seedance-2.0") == "volcengine/doubao-seedance-2.0"

    def test_normalize_bare_model_adds_volcengine_prefix(self):
        assert _normalize_pricing_model("doubao-seedance-2.0") == "volcengine/doubao-seedance-2.0"

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
