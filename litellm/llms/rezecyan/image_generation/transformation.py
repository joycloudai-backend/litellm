"""
Rezecyan Image Generation Config

Rezecyan exposes an OpenAI-compatible /v1/images/generations endpoint:
request and response shapes match OpenAI's
{"created": ts, "data": [{"url" | "b64_json": ...}]}.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import REZECYAN_DEFAULT_API_BASE, RezecyanError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class RezecyanImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "size",
            "quality",
            "response_format",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        if drop_params:
            non_default_params = {
                k: v for k, v in non_default_params.items() if k in supported_params
            }
        optional_params.update(non_default_params)
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = (
            api_base
            or get_secret_str("REZECYAN_API_BASE")
            or REZECYAN_DEFAULT_API_BASE
        )
        base = base.rstrip("/")
        if base.endswith("/images/generations"):
            return base
        return f"{base}/images/generations"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        final_api_key = api_key or get_secret_str("REZECYAN_API_KEY")
        if not final_api_key:
            raise ValueError(
                "Rezecyan API key is required. Set 'REZECYAN_API_KEY' env var or pass api_key explicitly."
            )
        default_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {final_api_key}",
        }
        return {**headers, **default_headers}

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        request_data: dict = {
            "model": model,
            "prompt": prompt,
        }
        for key, value in optional_params.items():
            if key.startswith("_"):
                continue
            request_data[key] = value
        return request_data

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse Rezecyan image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_data,
        )

        if "error" in response_data:
            error = response_data["error"]
            message = (
                error.get("message", str(error))
                if isinstance(error, dict)
                else str(error)
            )
            raise self.get_error_class(
                error_message=message,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        for item in response_data.get("data", []):
            model_response.data.append(
                ImageObject(
                    url=item.get("url"),
                    b64_json=item.get("b64_json"),
                    revised_prompt=item.get("revised_prompt"),
                )
            )

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        if isinstance(headers, dict):
            headers = httpx.Headers(headers)
        return RezecyanError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
