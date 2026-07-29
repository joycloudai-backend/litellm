"""
Unit tests for the Rezecyan embedding / rerank / image generation transformations.
"""

import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../.."))

import httpx
import pytest

from litellm.llms.rezecyan.common_utils import (
    REZECYAN_DEFAULT_API_BASE,
    RezecyanError,
)
from litellm.llms.rezecyan.embed.transformation import RezecyanEmbeddingConfig
from litellm.llms.rezecyan.image_generation.transformation import (
    RezecyanImageGenerationConfig,
)
from litellm.llms.rezecyan.rerank.transformation import RezecyanRerankConfig
from litellm.types.utils import EmbeddingResponse, ImageResponse


class TestRezecyanEmbedding:
    def test_complete_url(self):
        config = RezecyanEmbeddingConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key="k",
            model="text-embedding-v1",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{REZECYAN_DEFAULT_API_BASE}/embeddings"

    def test_validate_environment_requires_key(self, monkeypatch):
        monkeypatch.delenv("REZECYAN_API_KEY", raising=False)
        config = RezecyanEmbeddingConfig()
        with pytest.raises(ValueError, match="REZECYAN_API_KEY"):
            config.validate_environment(
                headers={},
                model="text-embedding-v1",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_transform_request_and_response(self):
        config = RezecyanEmbeddingConfig()
        req = config.transform_embedding_request(
            model="text-embedding-v1",
            input=["hello"],
            optional_params={"dimensions": 512, "encoding_format": None},
            headers={},
        )
        assert req == {"model": "text-embedding-v1", "input": ["hello"], "dimensions": 512}

        raw = MagicMock(spec=httpx.Response)
        raw.status_code = 200
        raw.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "text-embedding-v1",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        resp = config.transform_embedding_response(
            model="text-embedding-v1",
            raw_response=raw,
            model_response=EmbeddingResponse(),
            logging_obj=MagicMock(),
            api_key="k",
            request_data=req,
            optional_params={},
            litellm_params={},
        )
        assert resp.data[0]["embedding"] == [0.1, 0.2]
        assert resp.usage.prompt_tokens == 3

    def test_transform_response_error_body(self):
        config = RezecyanEmbeddingConfig()
        raw = MagicMock(spec=httpx.Response)
        raw.status_code = 400
        raw.json.return_value = {"error": {"message": "bad input"}}
        with pytest.raises(RezecyanError, match="bad input"):
            config.transform_embedding_response(
                model="text-embedding-v1",
                raw_response=raw,
                model_response=EmbeddingResponse(),
                logging_obj=MagicMock(),
                api_key="k",
                request_data={},
                optional_params={},
                litellm_params={},
            )


class TestRezecyanRerank:
    def test_complete_url(self):
        config = RezecyanRerankConfig()
        assert (
            config.get_complete_url(api_base=None, model="rerank-v1")
            == f"{REZECYAN_DEFAULT_API_BASE}/rerank"
        )
        # already-complete base is idempotent
        assert (
            config.get_complete_url(
                api_base="https://x.example.com/v1/rerank", model="rerank-v1"
            )
            == "https://x.example.com/v1/rerank"
        )

    def test_transform_request_requires_query_and_documents(self):
        config = RezecyanRerankConfig()
        with pytest.raises(ValueError, match="query"):
            config.transform_rerank_request(
                model="rerank-v1", optional_rerank_params={}, headers={}
            )
        req = config.transform_rerank_request(
            model="rerank-v1",
            optional_rerank_params={
                "query": "q",
                "documents": ["a", "b"],
                "top_n": 1,
            },
            headers={},
        )
        assert req == {
            "model": "rerank-v1",
            "query": "q",
            "documents": ["a", "b"],
            "top_n": 1,
        }

    def test_transform_response(self):
        config = RezecyanRerankConfig()
        raw = MagicMock(spec=httpx.Response)
        raw.status_code = 200
        raw.json.return_value = {
            "id": "rr-1",
            "results": [
                {"index": 1, "relevance_score": 0.9, "document": {"text": "b"}},
                {"index": 0, "relevance_score": 0.1},
            ],
            "usage": {"total_tokens": 12},
        }
        resp = config.transform_rerank_response(
            model="rerank-v1",
            raw_response=raw,
            model_response=MagicMock(),
            logging_obj=MagicMock(),
            api_key="k",
            request_data={"query": "q"},
        )
        assert resp.id == "rr-1"
        assert resp.results[0]["index"] == 1
        assert resp.results[0]["document"]["text"] == "b"
        assert resp.meta["billed_units"]["total_tokens"] == 12

    def test_transform_response_error_envelope(self):
        config = RezecyanRerankConfig()
        raw = MagicMock(spec=httpx.Response)
        raw.status_code = 400
        raw.json.return_value = {"code": "InvalidParameter", "message": "bad"}
        with pytest.raises(RezecyanError, match="bad"):
            config.transform_rerank_response(
                model="rerank-v1",
                raw_response=raw,
                model_response=MagicMock(),
                logging_obj=MagicMock(),
                api_key="k",
                request_data={},
            )


class TestRezecyanImageGeneration:
    def test_complete_url_and_request(self):
        config = RezecyanImageGenerationConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key="k",
            model="image-v1",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{REZECYAN_DEFAULT_API_BASE}/images/generations"

        req = config.transform_image_generation_request(
            model="image-v1",
            prompt="a cat",
            optional_params={"n": 2, "size": "1024x1024"},
            litellm_params={},
            headers={},
        )
        assert req == {
            "model": "image-v1",
            "prompt": "a cat",
            "n": 2,
            "size": "1024x1024",
        }

    def test_transform_response(self):
        config = RezecyanImageGenerationConfig()
        raw = MagicMock(spec=httpx.Response)
        raw.status_code = 200
        raw.json.return_value = {
            "created": 1,
            "data": [{"url": "https://img.example.com/1.png"}],
        }
        resp = config.transform_image_generation_response(
            model="image-v1",
            raw_response=raw,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={"prompt": "a cat"},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert resp.data[0].url == "https://img.example.com/1.png"

    def test_transform_response_error(self):
        config = RezecyanImageGenerationConfig()
        raw = MagicMock(spec=httpx.Response)
        raw.status_code = 400
        raw.headers = httpx.Headers()
        raw.json.return_value = {"error": {"message": "unsupported size"}}
        with pytest.raises(RezecyanError, match="unsupported size"):
            config.transform_image_generation_response(
                model="image-v1",
                raw_response=raw,
                model_response=ImageResponse(),
                logging_obj=MagicMock(),
                request_data={"prompt": "a cat"},
                optional_params={},
                litellm_params={},
                encoding=None,
            )
