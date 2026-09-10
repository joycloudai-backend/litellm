"""
Common utilities for BytePlus LLM provider (ByteDance international platform).
"""

from typing import Optional

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class BytePlusError(BaseLLMException):
    """
    Custom exception class for BytePlus provider errors.
    """

    def __init__(
        self, status_code: int, message: str, headers: Optional[httpx.Headers] = None
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers or httpx.Headers()
        super().__init__(
            status_code=status_code, message=message, headers=dict(self.headers)
        )


def get_byteplus_base_url(api_base: Optional[str] = None) -> str:
    """
    Get the base URL for BytePlus API calls.

    Args:
        api_base: Optional custom API base URL

    Returns:
        The base URL to use for API calls
    """
    if api_base:
        return api_base
    return "https://ark.ap-southeast.bytepluses.com"


def get_byteplus_headers(api_key: str, extra_headers: Optional[dict] = None) -> dict:
    """
    Get headers for BytePlus API calls.

    Args:
        api_key: The API key for authentication
        extra_headers: Optional additional headers

    Returns:
        Dictionary of headers
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if extra_headers:
        headers.update(extra_headers)

    return headers
