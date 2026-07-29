"""
Common utilities for the Rezecyan LLM provider.
"""

from typing import Optional

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

REZECYAN_DEFAULT_API_BASE = "https://www.rezecyan.com/v1"


class RezecyanError(BaseLLMException):
    """Exception class for Rezecyan provider errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers or httpx.Headers()
        super().__init__(
            status_code=status_code,
            message=message,
            headers=dict(self.headers),
        )
