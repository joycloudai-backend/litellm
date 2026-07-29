"""
Unit tests for the Rezecyan chat transformation.

Covers provider resolution, URL construction, DeepSeek-family
reasoning_content passback, and price-map cost calculation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

import litellm
from litellm.llms.rezecyan.chat.transformation import RezecyanChatConfig
from litellm.llms.rezecyan.common_utils import REZECYAN_DEFAULT_API_BASE


class TestRezecyanProviderResolution:
    def test_get_llm_provider_prefix(self):
        model, provider, _, api_base = litellm.get_llm_provider(
            "rezecyan/qwen3.7-plus"
        )
        assert model == "qwen3.7-plus"
        assert provider == "rezecyan"
        assert api_base == REZECYAN_DEFAULT_API_BASE

    def test_listed_in_openai_compatible_providers(self):
        assert "rezecyan" in litellm.openai_compatible_providers

    def test_provider_info_precedence(self, monkeypatch):
        config = RezecyanChatConfig()

        # explicit args win
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://custom.example.com/v1", "direct-key"
        )
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "direct-key"

        # env vars next
        monkeypatch.setenv("REZECYAN_API_BASE", "https://env.example.com/v1")
        monkeypatch.setenv("REZECYAN_API_KEY", "env-key")
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://env.example.com/v1"
        assert api_key == "env-key"

        # defaults last
        monkeypatch.delenv("REZECYAN_API_BASE")
        monkeypatch.delenv("REZECYAN_API_KEY")
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == REZECYAN_DEFAULT_API_BASE
        assert api_key is None

    def test_get_complete_url(self):
        config = RezecyanChatConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key="k",
            model="qwen3.7-plus",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{REZECYAN_DEFAULT_API_BASE}/chat/completions"

        # idempotent when base already targets the endpoint
        url = config.get_complete_url(
            api_base=url,
            api_key="k",
            model="qwen3.7-plus",
            optional_params={},
            litellm_params={},
        )
        assert url.count("/chat/completions") == 1


class TestRezecyanReasoningPassback:
    def test_requires_reasoning_passback_only_for_deepseek(self):
        assert RezecyanChatConfig._requires_reasoning_passback("deepseek-v3.2")
        assert RezecyanChatConfig._requires_reasoning_passback("DeepSeek-R2")
        assert not RezecyanChatConfig._requires_reasoning_passback("qwen3.7-plus")

    def test_fill_reasoning_content_promotes_from_provider_fields(self):
        config = RezecyanChatConfig()
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello",
                "provider_specific_fields": {"reasoning_content": "thought..."},
            },
            {"role": "user", "content": "again"},
        ]
        result = config._fill_reasoning_content(messages)  # type: ignore
        assert result[1]["reasoning_content"] == "thought..."
        assert "reasoning_content" not in result[1]["provider_specific_fields"]
        # non-assistant messages untouched
        assert result[0] == messages[0]
        assert result[2] == messages[2]

    def test_fill_reasoning_content_injects_placeholder(self):
        config = RezecyanChatConfig()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = config._fill_reasoning_content(messages)  # type: ignore
        assert result[1]["reasoning_content"] == " "

    def test_transform_request_fills_reasoning_for_deepseek_only(self):
        config = RezecyanChatConfig()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]

        deepseek_req = config.transform_request(
            model="deepseek-v3.2",
            messages=[dict(m) for m in messages],  # type: ignore
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert deepseek_req["messages"][1]["reasoning_content"] == " "

        qwen_req = config.transform_request(
            model="qwen3.7-plus",
            messages=[dict(m) for m in messages],  # type: ignore
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "reasoning_content" not in qwen_req["messages"][1]

    @pytest.mark.asyncio
    async def test_async_transform_request_fills_reasoning(self):
        config = RezecyanChatConfig()
        req = await config.async_transform_request(
            model="deepseek-v3.2",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],  # type: ignore
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert req["messages"][1]["reasoning_content"] == " "


class TestRezecyanPricing:
    def test_price_map_entry_and_cost(self):
        info = litellm.get_model_info("rezecyan/qwen3.7-plus")
        # 门市价 ¥2/¥8 每 1M tokens，按 7.2 汇率换算
        assert info["input_cost_per_token"] == pytest.approx(2.0 / 7.2 / 1e6, rel=1e-4)
        assert info["output_cost_per_token"] == pytest.approx(
            8.0 / 7.2 / 1e6, rel=1e-4
        )
        assert info.get("cache_read_input_token_cost") == pytest.approx(
            0.2 / 7.2 / 1e6, rel=1e-4
        )

    def test_completion_cost_matches_real_bill(self):
        """真实账单回归：11 prompt + 427 completion 应等于门市价 ¥0.003438。"""
        from litellm import completion_cost
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        resp = ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="你好"))],
            usage=Usage(prompt_tokens=11, completion_tokens=427, total_tokens=438),
            model="qwen3.7-plus",
        )
        cost_usd = completion_cost(
            completion_response=resp, model="rezecyan/qwen3.7-plus"
        )
        assert cost_usd * 7.2 == pytest.approx(0.003438, rel=1e-3)
