"""
BytePlus video generation API transformation.

This module provides the BytePlusVideoConfig class for transforming
video generation requests and responses between OpenAI format and
BytePlus (Ark international) API format.

BytePlus uses the same API as Volcengine but with a different base URL
and API key environment variable.
"""

from typing import Any, Optional, Union

import httpx

from litellm.secret_managers.main import get_secret_str

from ..common_utils import BytePlusError, get_byteplus_base_url, get_byteplus_headers
from ...volcengine.videos.transformation import VolcEngineVideoConfig


class BytePlusVideoConfig(VolcEngineVideoConfig):
    """
    BytePlus (Ark international) video generation task API.

    Inherits from VolcEngineVideoConfig since the API is identical,
    only differing in base URL, API key environment variable, and
    error class.
    """

    def __init__(self) -> None:
        super().__init__()

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[Any] = None,
    ) -> dict:
        if litellm_params and getattr(litellm_params, "api_key", None):
            api_key = api_key or litellm_params.api_key

        api_key = (
            api_key
            or get_secret_str("BYTEPLUS_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "BytePlus API key is required. Set BYTEPLUS_API_KEY environment variable or pass api_key."
            )

        return get_byteplus_headers(api_key=api_key, extra_headers=headers)

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base_url = (
            api_base
            or get_secret_str("BYTEPLUS_API_BASE")
            or get_byteplus_base_url()
        )
        base_url = base_url.rstrip("/")

        if base_url.endswith("/contents/generations/tasks"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/contents/generations/tasks"
        return f"{base_url}/api/v3/contents/generations/tasks"

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BytePlusError:
        typed_headers = (
            headers
            if isinstance(headers, httpx.Headers)
            else httpx.Headers(headers or {})
        )
        return BytePlusError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )

    def _normalize_model(self, model: str) -> str:
        if model.startswith("byteplus/"):
            return model.split("/", 1)[1]
        return model
