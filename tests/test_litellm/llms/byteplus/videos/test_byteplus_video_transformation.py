"""
Tests for BytePlus video generation transformation.
"""

from unittest.mock import Mock, patch

import httpx
import pytest

from litellm.llms.byteplus.common_utils import (
    BytePlusError,
    get_byteplus_base_url,
    get_byteplus_headers,
)
from litellm.llms.byteplus.videos.transformation import BytePlusVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestBytePlusCommonUtils:
    """Test BytePlus common utility functions."""

    def test_get_byteplus_base_url_default(self):
        assert get_byteplus_base_url() == "https://ark.ap-southeast.bytepluses.com"

    def test_get_byteplus_base_url_custom(self):
        custom = "https://custom.byteplus.com"
        assert get_byteplus_base_url(custom) == custom

    def test_get_byteplus_headers(self):
        headers = get_byteplus_headers(api_key="test-key-123")
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_get_byteplus_headers_with_extra(self):
        headers = get_byteplus_headers(
            api_key="test-key", extra_headers={"X-Custom": "value"}
        )
        assert headers["X-Custom"] == "value"
        assert headers["Authorization"] == "Bearer test-key"

    def test_byteplus_error(self):
        error = BytePlusError(status_code=401, message="Unauthorized")
        assert error.status_code == 401
        assert error.message == "Unauthorized"


class TestBytePlusVideoTransformation:
    """Test BytePlus video request / response transformations."""

    def setup_method(self):
        self.config = BytePlusVideoConfig()
        self.mock_logging_obj = Mock()

    def test_inherits_from_volcengine(self):
        from litellm.llms.volcengine.videos.transformation import VolcEngineVideoConfig

        assert isinstance(self.config, VolcEngineVideoConfig)

    def test_validate_environment_with_api_key_param(self):
        headers = self.config.validate_environment(
            headers={}, model="byteplus/dreamina-seedance-2-0-260128", api_key="ark-test-key"
        )
        assert headers["Authorization"] == "Bearer ark-test-key"

    @patch("litellm.llms.byteplus.videos.transformation.get_secret_str")
    def test_validate_environment_with_env_var(self, mock_get_secret):
        mock_get_secret.return_value = "env-test-key"
        headers = self.config.validate_environment(
            headers={}, model="byteplus/dreamina-seedance-2-0-260128"
        )
        assert headers["Authorization"] == "Bearer env-test-key"

    def test_validate_environment_missing_key_raises(self):
        with patch(
            "litellm.llms.byteplus.videos.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="BYTEPLUS_API_KEY"):
                self.config.validate_environment(
                    headers={}, model="byteplus/dreamina-seedance-2-0-260128"
                )

    def test_get_complete_url_default(self):
        url = self.config.get_complete_url(
            model="byteplus/dreamina-seedance-2-0-260128",
            api_base=None,
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"

    def test_get_complete_url_custom_base(self):
        url = self.config.get_complete_url(
            model="byteplus/test",
            api_base="https://custom.byteplus.com/api/v3",
            litellm_params={},
        )
        assert url == "https://custom.byteplus.com/api/v3/contents/generations/tasks"

    def test_get_complete_url_already_complete(self):
        url = self.config.get_complete_url(
            model="byteplus/test",
            api_base="https://custom.byteplus.com/api/v3/contents/generations/tasks",
            litellm_params={},
        )
        assert url == "https://custom.byteplus.com/api/v3/contents/generations/tasks"

    @patch("litellm.llms.byteplus.videos.transformation.get_secret_str")
    def test_get_complete_url_from_env(self, mock_get_secret):
        mock_get_secret.return_value = "https://env.byteplus.com/api/v3"
        url = self.config.get_complete_url(
            model="byteplus/test", api_base=None, litellm_params={}
        )
        assert url == "https://env.byteplus.com/api/v3/contents/generations/tasks"

    def test_normalize_model_strips_byteplus_prefix(self):
        assert self.config._normalize_model("byteplus/dreamina-seedance-2-0-260128") == "dreamina-seedance-2-0-260128"

    def test_normalize_model_no_prefix(self):
        assert self.config._normalize_model("dreamina-seedance-2-0-260128") == "dreamina-seedance-2-0-260128"

    def test_normalize_model_does_not_strip_volcengine_prefix(self):
        assert self.config._normalize_model("volcengine/doubao-seedance") == "volcengine/doubao-seedance"

    def test_get_error_class_returns_byteplus_error(self):
        error = self.config.get_error_class(
            error_message="test error", status_code=400, headers={}
        )
        assert isinstance(error, BytePlusError)
        assert error.status_code == 400
        assert error.message == "test error"

    def test_transform_video_create_request_builds_content_payload(self):
        data, files, url = self.config.transform_video_create_request(
            model="byteplus/dreamina-seedance-2-0-260128",
            prompt="A cat walking in a garden",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks",
            video_create_optional_request_params={
                "ratio": "16:9",
                "duration": 4,
                "resolution": "720p",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["model"] == "dreamina-seedance-2-0-260128"
        assert data["ratio"] == "16:9"
        assert data["duration"] == 4
        assert data["resolution"] == "720p"
        assert data["content"][0] == {
            "type": "text",
            "text": "A cat walking in a garden",
        }

    def test_map_openai_params(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": "11",
                "size": "1280x720",
                "generate_audio": True,
                "watermark": False,
            },
            model="byteplus/dreamina-seedance-2-0-260128",
            drop_params=False,
        )
        assert mapped["duration"] == 11
        assert mapped["ratio"] == "16:9"
        assert mapped["generate_audio"] is True
        assert mapped["watermark"] is False


class TestBytePlusVideoConfigDispatch:
    """Test that BytePlusVideoConfig is correctly dispatched."""

    def test_provider_config_manager_returns_byteplus_config(self):
        config = ProviderConfigManager.get_provider_video_config(
            model="byteplus/dreamina-seedance-2-0-260128",
            provider=LlmProviders.BYTEPLUS,
        )
        assert config is not None
        assert isinstance(config, BytePlusVideoConfig)

    def test_provider_enum_has_byteplus(self):
        assert LlmProviders.BYTEPLUS == "byteplus"
