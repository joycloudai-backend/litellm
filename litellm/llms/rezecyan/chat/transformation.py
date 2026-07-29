"""
Translates from OpenAI's `/v1/chat/completions` to Rezecyan's `/v1/chat/completions`.

Rezecyan is an OpenAI-compatible aggregator (hosts Qwen / DeepSeek / third-party
models behind a single endpoint). Responses may carry `message.reasoning_content`
and `usage.completion_tokens_details.reasoning_tokens`, both of which are parsed
natively by the OpenAI transformation.
"""

from typing import List, Optional, Tuple, cast

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import REZECYAN_DEFAULT_API_BASE


class RezecyanChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("REZECYAN_API_BASE")
            or REZECYAN_DEFAULT_API_BASE
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("REZECYAN_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if not api_base:
            api_base = REZECYAN_DEFAULT_API_BASE

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base

    @staticmethod
    def _requires_reasoning_passback(model: str) -> bool:
        # ponytail: DeepSeek-family models hosted on Rezecyan require
        # `reasoning_content` to be passed back on assistant messages in
        # multi-turn thinking conversations (same constraint as the native
        # DeepSeek API). Name-based heuristic; extend if Rezecyan onboards
        # other thinking-mode models with the same requirement.
        return "deepseek" in model.lower()

    def _fill_reasoning_content(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        For each assistant message missing `reasoning_content`:
          1. Promote it from `provider_specific_fields["reasoning_content"]`
             (where LiteLLM stores provider-specific response fields).
          2. Otherwise inject a single space — the minimum the API accepts.
        """
        result: List[AllMessageValues] = []
        for msg in messages:
            if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
                patched = dict(cast(dict, msg))
                provider_fields = patched.get("provider_specific_fields") or {}
                stored = provider_fields.get("reasoning_content")
                if stored:
                    patched["reasoning_content"] = stored
                    cleaned = dict(provider_fields)
                    cleaned.pop("reasoning_content", None)
                    patched["provider_specific_fields"] = cleaned
                else:
                    litellm.verbose_logger.warning(
                        "Rezecyan thinking mode: assistant message is missing "
                        "`reasoning_content`; injecting a single-space placeholder "
                        "to satisfy API validation. Preserve `reasoning_content` "
                        "from the original assistant response when building "
                        "multi-turn conversation history."
                    )
                    patched["reasoning_content"] = " "
                result.append(cast(AllMessageValues, patched))
            else:
                result.append(msg)
        return result

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        if self._requires_reasoning_passback(model):
            messages = self._fill_reasoning_content(messages)
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        if self._requires_reasoning_passback(model):
            messages = self._fill_reasoning_content(messages)
        return await super().async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
