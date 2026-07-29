"""
Transformation logic for Rezecyan's /v1/rerank API.

Endpoint
- https://www.rezecyan.com/v1/rerank

Request fields: model, query, documents, top_n, return_documents.
Response: results[].index, results[].relevance_score, optionally
results[].document.text (when return_documents=true), plus a top-level
usage.total_tokens counter.
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)

from ..common_utils import REZECYAN_DEFAULT_API_BASE, RezecyanError


class RezecyanRerankConfig(BaseRerankConfig):
    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        base = (
            api_base
            or get_secret_str("REZECYAN_API_BASE")
            or REZECYAN_DEFAULT_API_BASE
        )
        base = base.rstrip("/")
        if base.endswith("/rerank"):
            return base
        return f"{base}/rerank"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("REZECYAN_API_KEY")
        if api_key is None:
            raise ValueError(
                "Rezecyan API key is required. Set 'REZECYAN_API_KEY' env var or pass api_key explicitly."
            )

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        return {**default_headers, **headers}

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return ["query", "documents", "top_n", "return_documents"]

    def map_cohere_rerank_params(
        self,
        non_default_params: Optional[dict],
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
    ) -> Dict:
        # Accepts query/documents/top_n/return_documents; the rest
        # (rank_fields, max_*_per_doc) are silently dropped.
        params: OptionalRerankParams = OptionalRerankParams(
            query=query,
            documents=documents,
        )
        if top_n is not None:
            params["top_n"] = top_n
        if return_documents is not None:
            params["return_documents"] = return_documents
        return dict(params)

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
        litellm_params: Optional[dict] = None,
    ) -> dict:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Rezecyan rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Rezecyan rerank")

        request: Dict[str, Any] = {
            "model": model,
            "query": optional_rerank_params["query"],
            "documents": optional_rerank_params["documents"],
        }
        if optional_rerank_params.get("top_n") is not None:
            request["top_n"] = optional_rerank_params["top_n"]
        if optional_rerank_params.get("return_documents") is not None:
            request["return_documents"] = optional_rerank_params["return_documents"]
        return request

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: Optional[dict] = None,
        optional_params: Optional[dict] = None,
        litellm_params: Optional[dict] = None,
    ) -> RerankResponse:
        request_data = request_data or {}
        optional_params = optional_params or {}
        litellm_params = litellm_params or {}
        try:
            response_json = raw_response.json()
        except Exception:
            raise RezecyanError(
                status_code=raw_response.status_code,
                message=raw_response.text,
            )

        logging_obj.post_call(
            input=request_data.get("query"),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )

        if "code" in response_json and "results" not in response_json:
            raise RezecyanError(
                status_code=raw_response.status_code,
                message=response_json.get("message", str(response_json)),
            )

        results = response_json.get("results")
        if results is None:
            raise RezecyanError(
                status_code=raw_response.status_code,
                message=f"No results in Rezecyan rerank response: {response_json}",
            )

        transformed_results: List[dict] = []
        for r in results:
            item: Dict[str, Any] = {
                "index": r["index"],
                "relevance_score": r["relevance_score"],
            }
            doc = r.get("document")
            if isinstance(doc, dict):
                item["document"] = doc
            elif isinstance(doc, str):
                # Defensive: normalize string-shaped document echoes.
                item["document"] = {"text": doc}
            transformed_results.append(item)

        usage = response_json.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        billed_units = RerankBilledUnits(total_tokens=total_tokens)
        tokens = RerankTokens(input_tokens=total_tokens)
        meta = RerankResponseMeta(billed_units=billed_units, tokens=tokens)

        return RerankResponse(
            id=response_json.get("id") or str(uuid.uuid4()),
            results=transformed_results,  # type: ignore
            meta=meta,
        )

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
