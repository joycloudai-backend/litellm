"""
Translates from OpenAI's `/v1/chat/completions` to Rezecyan's `/v1/chat/completions`.

Rezecyan is an OpenAI-compatible aggregator (hosts Qwen / DeepSeek / third-party
models behind a single endpoint). Responses may carry `message.reasoning_content`
and `usage.completion_tokens_details.reasoning_tokens`. Thinking models that
set ``text_tokens == completion_tokens`` while also reporting reasoning as a
subset are clamped before LiteLLM's generic cost calculator, so reasoning is
not billed twice. Consistent OpenAI-style breakdowns are left unchanged.
"""

from typing import Any, List, Optional, Tuple, cast

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

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

    @staticmethod
    def _clamp_overcounted_reasoning_text_tokens(usage: Any) -> None:
        """Fix Reze thinking-model usage so LiteLLM does not double-bill reasoning.

        Some Reze models report ``text_tokens == completion_tokens`` while
        ``reasoning_tokens`` is already a subset of ``completion_tokens``
        (``total_tokens == prompt + completion``). LiteLLM's generic cost
        calculator then charges ``text + reasoning`` at the output rate.

        If ``text + reasoning > completion``, rewrite
        ``text_tokens = max(0, completion_tokens - reasoning_tokens)``.
        Leaves consistent OpenAI-style breakdowns untouched. In-place.
        """
        if usage is None:
            return

        if isinstance(usage, dict):
            details = usage.get("completion_tokens_details")
        else:
            details = getattr(usage, "completion_tokens_details", None)
        if details is None:
            return

        def _field(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        try:
            reasoning_tokens = int(_field(details, "reasoning_tokens", 0) or 0)
            text_raw = _field(details, "text_tokens", None)
            if text_raw is None:
                return
            text_tokens = int(text_raw)
            completion_tokens = int(_field(usage, "completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            return

        if reasoning_tokens <= 0 or text_tokens + reasoning_tokens <= completion_tokens:
            return

        new_text = max(0, completion_tokens - reasoning_tokens)
        if isinstance(details, dict):
            details["text_tokens"] = new_text
        else:
            details.text_tokens = new_text

    def transform_parsed_response_dict(self, parsed_response: dict) -> dict:
        parsed_response = super().transform_parsed_response_dict(parsed_response)
        if isinstance(parsed_response, dict):
            self._clamp_overcounted_reasoning_text_tokens(parsed_response.get("usage"))
        return parsed_response

    def transform_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        response = super().transform_response(*args, **kwargs)
        self._clamp_overcounted_reasoning_text_tokens(getattr(response, "usage", None))
        return response

    def apply_assembled_streaming_response_metadata(
        self, response: ModelResponse, chunks: List[Any]
    ) -> None:
        super().apply_assembled_streaming_response_metadata(response, chunks)
        self._clamp_overcounted_reasoning_text_tokens(getattr(response, "usage", None))

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
