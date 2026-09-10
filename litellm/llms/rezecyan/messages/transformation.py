"""
Rezecyan Anthropic-compatible messages transformation config.

Routes Claude Code / Anthropic ``/v1/messages`` traffic to Rezecyan's native
Messages endpoint instead of converting through OpenAI chat completions.

OpenAI ``/v1/chat/completions`` continues to use ``RezecyanChatConfig``.
"""

from typing import Any, List, Optional, Tuple

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str

from ..common_utils import REZECYAN_DEFAULT_API_BASE


class RezecyanAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    Rezecyan exposes an Anthropic-compatible Messages API at
    ``https://api.rezecyan.com/v1/messages`` (same host as chat completions).

    Using this config avoids LiteLLM's Messages→ChatCompletions→Messages
    streaming adapter, which can emit invalid Anthropic SSE for models that
    return OpenAI-style ``reasoning_content`` (e.g. kimi-k2.7-code).
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "rezecyan"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("REZECYAN_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        return (
            api_base
            or get_secret_str("REZECYAN_ANTHROPIC_API_BASE")
            or get_secret_str("REZECYAN_API_BASE")
            or REZECYAN_DEFAULT_API_BASE
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        dynamic_api_key = self.get_api_key(api_key=api_key)

        # Rezecyan (NewAPI) accepts Bearer tokens on /v1/messages.
        has_auth = any(k.lower() in ("authorization", "x-api-key") for k in headers)
        if not has_auth and dynamic_api_key is not None:
            headers["Authorization"] = f"Bearer {dynamic_api_key}"

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
            custom_llm_provider=self.custom_llm_provider or "rezecyan",
        )

        return headers, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url = self.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith("/v1/messages"):
            return base_url

        # Deployments often store the OpenAI chat base (.../v1 or .../chat/completions).
        if base_url.endswith("/v1/chat/completions"):
            base_url = base_url[: -len("/v1/chat/completions")]
        elif base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        elif base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]

        return f"{base_url}/v1/messages"
