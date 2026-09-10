"""
Unit tests for Rezecyan Anthropic Messages config.

Ensures /v1/messages traffic is routed to Rezecyan's native Messages endpoint
instead of the Messages→ChatCompletions conversion path.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.rezecyan.common_utils import REZECYAN_DEFAULT_API_BASE
from litellm.llms.rezecyan.messages.transformation import (
    RezecyanAnthropicMessagesConfig,
)
from litellm.utils import ProviderConfigManager


def test_rezecyan_provider_uses_anthropic_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="kimi-k2.7-code",
        provider=litellm.LlmProviders.REZECYAN,
    )

    assert isinstance(config, RezecyanAnthropicMessagesConfig)
    assert config.custom_llm_provider == "rezecyan"


def test_anthropic_provider_keeps_default_config_for_kimi_named_model():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="kimi-k2.7-code",
        provider=litellm.LlmProviders.ANTHROPIC,
    )

    assert isinstance(config, AnthropicMessagesConfig)
    assert not isinstance(config, RezecyanAnthropicMessagesConfig)


def test_strips_billing_metadata():
    config = RezecyanAnthropicMessagesConfig()
    assert config.should_strip_billing_metadata() is True


def test_get_api_base_default():
    config = RezecyanAnthropicMessagesConfig()
    assert config.get_api_base() == REZECYAN_DEFAULT_API_BASE


def test_get_api_base_from_arg():
    config = RezecyanAnthropicMessagesConfig()
    assert (
        config.get_api_base(api_base="https://custom.example.com/v1")
        == "https://custom.example.com/v1"
    )


def test_messages_url_default():
    config = RezecyanAnthropicMessagesConfig()
    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="kimi-k2.7-code",
            optional_params={},
            litellm_params={},
        )
        == f"{REZECYAN_DEFAULT_API_BASE}/messages"
    )


def test_messages_url_with_base_ending_in_v1():
    config = RezecyanAnthropicMessagesConfig()
    assert (
        config.get_complete_url(
            api_base="https://api.rezecyan.com/v1",
            api_key=None,
            model="kimi-k2.7-code",
            optional_params={},
            litellm_params={},
        )
        == "https://api.rezecyan.com/v1/messages"
    )


def test_messages_url_with_base_ending_in_v1_messages():
    config = RezecyanAnthropicMessagesConfig()
    url = config.get_complete_url(
        api_base="https://api.rezecyan.com/v1/messages",
        api_key=None,
        model="kimi-k2.7-code",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.rezecyan.com/v1/messages"


def test_messages_url_with_base_ending_in_chat_completions():
    config = RezecyanAnthropicMessagesConfig()
    assert (
        config.get_complete_url(
            api_base="https://api.rezecyan.com/v1/chat/completions",
            api_key=None,
            model="kimi-k2.7-code",
            optional_params={},
            litellm_params={},
        )
        == "https://api.rezecyan.com/v1/messages"
    )


def test_messages_url_with_custom_host_no_v1():
    config = RezecyanAnthropicMessagesConfig()
    assert (
        config.get_complete_url(
            api_base="https://api.rezecyan.com",
            api_key=None,
            model="kimi-k2.7-code",
            optional_params={},
            litellm_params={},
        )
        == "https://api.rezecyan.com/v1/messages"
    )


def test_validate_environment_sets_bearer_auth():
    config = RezecyanAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="kimi-k2.7-code",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-reze-key",
        api_base="https://custom.test/v1",
    )

    assert headers["Authorization"] == "Bearer sk-reze-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert api_base == "https://custom.test/v1"


def test_validate_environment_preserves_existing_authorization():
    config = RezecyanAnthropicMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={"Authorization": "Bearer existing", "anthropic-version": "2024-01-01"},
        model="kimi-k2.7-code",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-reze-key",
        api_base=None,
    )

    assert headers["Authorization"] == "Bearer existing"
    assert headers["anthropic-version"] == "2024-01-01"
