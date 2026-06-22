"""
Volcengine video billing module.

This module provides the VolcengineVideoBillingManager class for accurate
async token-based billing for Volcengine video generation. It handles:
- Pending video task registration
- Status polling and final cost reconciliation
- Async billing delta application
- Spend log management with provider pricing currency conversion
"""

import hashlib
import json
import os
import re
import traceback
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.volcengine.videos.transformation import VolcEngineVideoConfig
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders, StandardLoggingPayload
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import extract_original_video_id

if TYPE_CHECKING:
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router
else:
    PrismaClient = Any
    ProxyLogging = Any
    Router = Any
    DBSpendUpdateWriter = Any


VOLCENGINE_VIDEO_CREATE_CALL_TYPES = {
    "avideo_generation",
    "acreate_video",
    "create_video",
}
VOLCENGINE_VIDEO_STATUS_CALL_TYPES = {
    "avideo_status",
    "avideo_retrieve",
    "video_retrieve",
}
VOLCENGINE_VIDEO_CONTENT_CALL_TYPES = {
    "avideo_content",
    "video_content",
}
VOLCENGINE_VIDEO_SUCCESS_CALL_TYPES = (
    VOLCENGINE_VIDEO_CREATE_CALL_TYPES
    | VOLCENGINE_VIDEO_STATUS_CALL_TYPES
    | VOLCENGINE_VIDEO_CONTENT_CALL_TYPES
)
VOLCENGINE_VIDEO_ZERO_COST_CALL_TYPES = VOLCENGINE_VIDEO_SUCCESS_CALL_TYPES
VOLCENGINE_VIDEO_PENDING_STATUSES = {"queued", "processing"}
VOLCENGINE_VIDEO_NO_CHARGE_STATUSES = {"failed", "cancelled", "expired", "deleted"}
VOLCENGINE_VIDEO_COMPLETED_STATUS = "completed"
VOLCENGINE_VIDEO_DEFAULT_PRICING_MODEL = "volcengine/doubao-seedance-2.0"
VOLCENGINE_VIDEO_OUTPUT_COST_KEY_PREFIX = (
    "volcengine_video_output_cost_per_million_tokens"
)
VOLCENGINE_VIDEO_POLL_INTERVAL_SECONDS = 15
VOLCENGINE_VIDEO_RETRY_INTERVAL_SECONDS = 60
VOLCENGINE_VIDEO_CNY_PER_USD_ENV = "LITELLM_VOLCENGINE_VIDEO_CNY_PER_USD"
VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD = 7.2

# BytePlus (international) video billing constants
BYTEPLUS_VIDEO_DEFAULT_PRICING_MODEL = "byteplus/dreamina-seedance-2.0"
ARK_VIDEO_PROVIDERS = {"volcengine", "byteplus"}
VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS: Dict[str, Dict[str, Any]] = {
    "volcengine/doubao-seedance-2.0": {
        "litellm_provider": "volcengine",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "CNY",
        "source": "https://www.volcengine.com/docs/82379/1544106?lang=zh",
        "supported_modalities": ["text", "image", "video", "audio"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_input_video": 46.0,
        "volcengine_video_output_cost_per_million_tokens_with_input_video": 28.0,
        "volcengine_video_output_cost_per_million_tokens_without_input_video_1080p": 51.0,
        "volcengine_video_output_cost_per_million_tokens_with_input_video_1080p": 31.0,
    },
    "volcengine/doubao-seedance-2.0-fast": {
        "litellm_provider": "volcengine",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "CNY",
        "source": "https://www.volcengine.com/docs/82379/1544106?lang=zh",
        "supported_modalities": ["text", "image", "video", "audio"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_input_video": 37.0,
        "volcengine_video_output_cost_per_million_tokens_with_input_video": 22.0,
    },
    "volcengine/seedance-1.5-pro": {
        "litellm_provider": "volcengine",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "CNY",
        "source": "https://www.volcengine.com/docs/82379/1366799",
        "supported_modalities": ["text", "image", "video", "audio"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_audio": 8.0,
        "volcengine_video_output_cost_per_million_tokens_with_audio": 16.0,
    },
    "byteplus/dreamina-seedance-2.0": {
        "litellm_provider": "byteplus",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "USD",
        "source": "https://www.byteplus.com/docs/82379/1544106",
        "supported_modalities": ["text", "image", "video", "audio"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_input_video": 7.0,
        "volcengine_video_output_cost_per_million_tokens_with_input_video": 4.3,
        "volcengine_video_output_cost_per_million_tokens_without_input_video_1080p": 7.7,
        "volcengine_video_output_cost_per_million_tokens_with_input_video_1080p": 4.7,
    },
    "byteplus/dreamina-seedance-2.0-fast": {
        "litellm_provider": "byteplus",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "USD",
        "source": "https://www.byteplus.com/docs/82379/1544106",
        "supported_modalities": ["text", "image", "video", "audio"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_input_video": 5.6,
        "volcengine_video_output_cost_per_million_tokens_with_input_video": 3.3,
    },
    "byteplus/seedance-1.5-pro": {
        "litellm_provider": "byteplus",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "USD",
        "source": "https://www.byteplus.com/docs/82379/1544106",
        "supported_modalities": ["text", "image", "video", "audio"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_audio": 1.2,
        "volcengine_video_output_cost_per_million_tokens_with_audio": 2.4,
    },
    "byteplus/seedance-1-0-pro": {
        "litellm_provider": "byteplus",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "USD",
        "source": "https://www.byteplus.com/docs/82379/1544106",
        "supported_modalities": ["text", "image"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_input_video": 2.5,
    },
}


def _entry_has_video_pricing(entry: Optional[Dict[str, Any]]) -> bool:
    if not entry:
        return False
    return any(
        key.startswith(VOLCENGINE_VIDEO_OUTPUT_COST_KEY_PREFIX)
        and entry.get(key) is not None
        for key in entry
    )


def register_ark_video_pricing_models() -> None:
    """
    Register the in-code Volcengine/BytePlus video pricing entries into
    litellm.model_cost so both async billing and /v1/model/info can resolve
    them. Idempotent: registers only when an entry is missing required keys.
    """
    for model_name, model_info in VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS.items():
        existing_model_info = litellm.model_cost.get(model_name) or {}
        required_keys = ["provider_pricing_currency"] + [
            key
            for key in model_info
            if key.startswith(VOLCENGINE_VIDEO_OUTPUT_COST_KEY_PREFIX)
        ]
        if not all(existing_model_info.get(key) is not None for key in required_keys):
            litellm.register_model(model_cost=VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS)
            verbose_proxy_logger.info(
                "Registered runtime pricing overrides for Volcengine video billing"
            )
            return


def get_ark_video_pricing_entry(
    model_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Resolve the Volcengine/BytePlus video pricing entry for a deployment's
    model_info, preferring provider_pricing_model over base_model and applying
    the same dotted/versionless candidate normalization the billing path uses
    (e.g. "volcengine/doubao-seedance-2-0-260128" -> "volcengine/doubao-seedance-2.0").

    Returns the litellm.model_cost entry carrying the
    volcengine_video_output_cost_per_million_tokens_* keys, or None when no
    pricing config matches.
    """
    pricing_model = model_info.get("provider_pricing_model") or model_info.get(
        "base_model"
    )
    if not pricing_model:
        return None
    register_ark_video_pricing_models()
    for candidate in _candidate_pricing_models(str(pricing_model)):
        entry = litellm.model_cost.get(candidate)
        if _entry_has_video_pricing(entry):
            return entry
    return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token_if_needed(token: Optional[str]) -> str:
    if token is None:
        return ""
    if token.startswith("sk-"):
        return hashlib.sha256(token.encode()).hexdigest()
    return token


def _normalize_pricing_model(model_name: str) -> str:
    if model_name.startswith("volcengine/") or model_name.startswith("byteplus/"):
        return model_name
    return f"volcengine/{model_name}"


def _candidate_pricing_models(model_name: str) -> List[str]:
    normalized_model = _normalize_pricing_model(model_name)
    provider, _, raw_model = normalized_model.partition("/")

    raw_variants = [raw_model]
    versionless_model = re.sub(r"-\d{6,}$", "", raw_model)
    if versionless_model and versionless_model != raw_model:
        raw_variants.append(versionless_model)

    # Volcengine model ids spell the version with a dash (e.g.
    # "doubao-seedance-2-0-260128") while the pricing keys use a dot
    # ("doubao-seedance-2.0"). Add dotted variants so the raw provider model
    # id resolves to its pricing entry without manual provider_pricing_model.
    for variant in list(raw_variants):
        dotted = re.sub(r"(\d)-(\d)", r"\1.\2", variant)
        if dotted != variant:
            raw_variants.append(dotted)

    candidates: List[str] = []
    for variant in raw_variants:
        candidate = f"{provider}/{variant}"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _is_1080p_resolution(resolution: Any) -> bool:
    if not resolution:
        return False
    normalized = str(resolution).strip().lower()
    if normalized in {"1080p", "1080"}:
        return True
    if "x" in normalized:
        try:
            dimensions = [int(part) for part in normalized.split("x")]
        except ValueError:
            return False
        return bool(dimensions) and min(dimensions) == 1080
    return False


def _has_reference_video(content: Any) -> bool:
    if not isinstance(content, list):
        return False

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        item_role = str(item.get("role") or "").lower()
        if item_type == "video_url" or item_role == "reference_video":
            return True
        if isinstance(item.get("video_url"), dict):
            return True
    return False


def _parse_request_tags(request_tags: Any) -> List[str]:
    if isinstance(request_tags, list):
        return [str(tag) for tag in request_tags if isinstance(tag, str) and tag]
    if isinstance(request_tags, str):
        parsed = safe_json_loads(request_tags, default=[])
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed if isinstance(tag, str) and tag]
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _ts_to_datetime(value: Optional[int]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _to_prisma_json(value: Any) -> Any:
    import prisma

    return prisma.Json(value)


def _normalize_non_empty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    string_value = str(value).strip()
    if not string_value:
        return None
    return string_value


def _get_cny_per_usd_rate() -> float:
    configured_rate = _safe_float(
        os.getenv(VOLCENGINE_VIDEO_CNY_PER_USD_ENV),
        default=VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD,
    )
    if configured_rate <= 0:
        return VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD
    return configured_rate


def _convert_provider_spend_to_usd(amount: float, currency: str) -> float:
    normalized_currency = (currency or "USD").upper()
    if normalized_currency == "USD":
        return amount
    if normalized_currency == "CNY":
        return amount / _get_cny_per_usd_rate()

    verbose_proxy_logger.warning(
        "Volcengine video billing unsupported provider currency=%s. Leaving spend unchanged.",
        normalized_currency,
    )
    return amount


class VolcengineVideoBillingManager:
    """
    Accurate async token-based billing for Volcengine video generation.

    Flow:
    1. Create request logs a zero-cost spend log and registers a pending video task.
    2. Status/content requests are always zero-cost request logs.
    3. When the task reaches a terminal status, the finalizer computes provider cost
       from provider-reported usage.total_tokens and updates spend tracking exactly once.
    """

    def __init__(
        self,
        prisma_client: "PrismaClient",
        llm_router: "Router",
        db_spend_update_writer: "DBSpendUpdateWriter",
        proxy_logging_obj: "ProxyLogging",
    ) -> None:
        self.prisma_client = prisma_client
        self.llm_router = llm_router
        self.db_spend_update_writer = db_spend_update_writer
        self.proxy_logging_obj = proxy_logging_obj
        self._video_task_table_unavailable = False
        self._pricing_models_registered = False

    def should_handle_success_event(self, kwargs: dict) -> bool:
        call_type = kwargs.get("call_type")
        if call_type not in VOLCENGINE_VIDEO_SUCCESS_CALL_TYPES:
            return False

        custom_llm_provider = kwargs.get("custom_llm_provider") or (
            kwargs.get("litellm_params", {}) or {}
        ).get("custom_llm_provider")
        return custom_llm_provider in ARK_VIDEO_PROVIDERS

    async def handle_success_event(
        self,
        kwargs: dict,
        completion_response: Optional[Any],
    ) -> Optional[float]:
        if not self.should_handle_success_event(kwargs):
            return None

        self._force_zero_cost_response(
            kwargs=kwargs, completion_response=completion_response
        )

        call_type = kwargs.get("call_type")
        video_response = self._coerce_video_object(completion_response)
        try:
            if (
                call_type in VOLCENGINE_VIDEO_CREATE_CALL_TYPES
                and video_response is not None
            ):
                await self._register_pending_video_task(
                    kwargs=kwargs,
                    completion_response=video_response,
                )
            elif (
                call_type in VOLCENGINE_VIDEO_STATUS_CALL_TYPES
                and video_response is not None
            ):
                await self._reconcile_task_from_video_response(
                    video_id=video_response.id or "",
                    video_response=video_response,
                    kwargs=kwargs,
                )
        except Exception as e:
            verbose_proxy_logger.error(
                "Volcengine video billing success-event hook failed: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
        return 0.0

    async def poll_pending_video_tasks(self) -> None:
        if self.prisma_client is None or self.llm_router is None:
            return

        video_task_table = self._get_video_task_table_model()
        if video_task_table is None:
            return

        now = _now_utc()
        tasks = await video_task_table.find_many(
            where={
                "billing_state": "pending",
                "OR": [
                    {"next_check_at": None},
                    {"next_check_at": {"lte": now}},
                ],
            },
            take=MAX_OBJECTS_PER_POLL_CYCLE,
            order={"created_at": "asc"},
        )

        for task in tasks:
            try:
                await self._poll_single_task(task=task)
            except Exception as e:
                verbose_proxy_logger.error(
                    "Volcengine video billing poll failed for task=%s: %s\n%s",
                    getattr(task, "video_id", None),
                    str(e),
                    traceback.format_exc(),
                )
                await video_task_table.update(
                    where={"video_id": task.video_id},
                    data={
                        "last_error": str(e),
                        "last_checked_at": now,
                        "next_check_at": now
                        + timedelta(seconds=VOLCENGINE_VIDEO_RETRY_INTERVAL_SECONDS),
                        "check_attempts": {"increment": 1},
                    },
                )

    def _force_zero_cost_response(
        self,
        kwargs: dict,
        completion_response: Optional[Any],
    ) -> None:
        kwargs["response_cost"] = 0.0
        standard_logging_object = cast(
            Optional[StandardLoggingPayload], kwargs.get("standard_logging_object")
        )
        if standard_logging_object is not None:
            standard_logging_object["response_cost"] = 0.0

        hidden_params = dict(getattr(completion_response, "_hidden_params", {}) or {})
        if hidden_params:
            hidden_params["response_cost"] = 0.0
            setattr(completion_response, "_hidden_params", hidden_params)

    def _coerce_video_object(
        self,
        completion_response: Optional[Any],
    ) -> Optional[VideoObject]:
        if completion_response is None:
            return None
        if isinstance(completion_response, VideoObject):
            return completion_response

        response_payload: Optional[Dict[str, Any]] = None
        if isinstance(completion_response, dict):
            response_payload = completion_response
        elif hasattr(completion_response, "model_dump"):
            try:
                model_dump = completion_response.model_dump()
                if isinstance(model_dump, dict):
                    response_payload = model_dump
            except Exception:
                response_payload = None
        elif hasattr(completion_response, "dict"):
            try:
                dict_payload = completion_response.dict()
                if isinstance(dict_payload, dict):
                    response_payload = dict_payload
            except Exception:
                response_payload = None

        if not isinstance(response_payload, dict):
            return None

        response_id = response_payload.get("id")
        response_object = response_payload.get("object")
        if response_id is None or response_object != "video":
            return None

        try:
            video_response = VideoObject(**response_payload)
        except Exception:
            return None

        hidden_params = getattr(completion_response, "_hidden_params", None)
        if hidden_params:
            video_response._hidden_params = dict(hidden_params)

        return video_response

    async def _register_pending_video_task(
        self,
        kwargs: dict,
        completion_response: VideoObject,
    ) -> None:
        video_task_table = self._get_video_task_table_model()
        if video_task_table is None:
            return

        video_id = completion_response.id or ""
        if not video_id:
            return

        metadata = get_litellm_metadata_from_kwargs(kwargs=kwargs)
        standard_logging_object = cast(
            Optional[StandardLoggingPayload], kwargs.get("standard_logging_object")
        )

        request_content = await self._get_request_content_for_task_registration(
            kwargs=kwargs,
            completion_response=completion_response,
        )
        has_input_video = _has_reference_video(request_content)
        resolution = self._extract_request_resolution(
            kwargs=kwargs, completion_response=completion_response
        )
        generate_audio = self._extract_request_generate_audio(
            kwargs=kwargs, completion_response=completion_response
        )

        model_info = cast(dict, metadata.get("model_info", {}) or {})
        pricing_model = self._resolve_pricing_model(model_info=model_info)
        unit_price, pricing_currency = self._resolve_pricing_snapshot(
            pricing_model=pricing_model,
            has_input_video=has_input_video,
            resolution=resolution,
            generate_audio=generate_audio,
        )

        api_key_hash = self._get_api_key_hash(
            metadata=metadata,
            standard_logging_object=standard_logging_object,
        )
        spend_log_identity = await self._get_spend_log_identity(video_id=video_id)
        request_tags = self._get_request_tags(
            metadata=metadata,
            standard_logging_object=standard_logging_object,
        )
        request_tags_json = _to_prisma_json(request_tags)
        custom_discount = self._snapshot_custom_discount(metadata)
        task_metadata_payload: Dict[str, Any] = {
            "request_content": request_content,
            "generate_audio": generate_audio,
        }
        if custom_discount is not None:
            task_metadata_payload["custom_discount"] = custom_discount
        task_metadata_json = _to_prisma_json(task_metadata_payload)
        now = _now_utc()
        usage = dict(completion_response.usage or {})

        await video_task_table.upsert(
            where={"video_id": video_id},
            data={
                "create": {
                    "video_id": video_id,
                    "provider_task_id": extract_original_video_id(video_id),
                    "api_key": spend_log_identity.get("api_key") or api_key_hash,
                    "user": spend_log_identity.get("user")
                    or metadata.get("user_api_key_user_id")
                    or "",
                    "team_id": spend_log_identity.get("team_id")
                    or metadata.get("user_api_key_team_id")
                    or None,
                    "organization_id": spend_log_identity.get("organization_id")
                    or metadata.get("user_api_key_org_id")
                    or None,
                    "end_user": spend_log_identity.get("end_user")
                    or metadata.get("user_api_key_end_user_id")
                    or None,
                    "custom_llm_provider": kwargs.get("custom_llm_provider") or "",
                    "model": kwargs.get("model") or "",
                    "model_group": metadata.get("model_group")
                    or kwargs.get("model")
                    or "",
                    "model_id": model_info.get("id") or "",
                    "provider_model": completion_response.model or "",
                    "pricing_model": pricing_model,
                    "pricing_currency": pricing_currency,
                    "price_per_million_tokens": unit_price,
                    "has_input_video": has_input_video,
                    "provider_status": completion_response.status or "queued",
                    "duration_seconds": _safe_float(
                        usage.get("duration_seconds"),
                        default=_safe_float(completion_response.seconds, 0.0),
                    )
                    or None,
                    "request_tags": request_tags_json,
                    "metadata": task_metadata_json,
                    "next_check_at": now
                    + timedelta(seconds=VOLCENGINE_VIDEO_POLL_INTERVAL_SECONDS),
                    "last_checked_at": now,
                    "check_attempts": 0,
                },
                "update": {
                    "api_key": spend_log_identity.get("api_key") or api_key_hash,
                    "user": spend_log_identity.get("user")
                    or metadata.get("user_api_key_user_id")
                    or "",
                    "team_id": spend_log_identity.get("team_id")
                    or metadata.get("user_api_key_team_id")
                    or None,
                    "organization_id": spend_log_identity.get("organization_id")
                    or metadata.get("user_api_key_org_id")
                    or None,
                    "end_user": spend_log_identity.get("end_user")
                    or metadata.get("user_api_key_end_user_id")
                    or None,
                    "custom_llm_provider": kwargs.get("custom_llm_provider") or "",
                    "model": kwargs.get("model") or "",
                    "model_group": metadata.get("model_group")
                    or kwargs.get("model")
                    or "",
                    "model_id": model_info.get("id") or "",
                    "provider_model": completion_response.model or "",
                    "pricing_model": pricing_model,
                    "pricing_currency": pricing_currency,
                    "price_per_million_tokens": unit_price,
                    "has_input_video": has_input_video,
                    "provider_status": completion_response.status or "queued",
                    "duration_seconds": _safe_float(
                        usage.get("duration_seconds"),
                        default=_safe_float(completion_response.seconds, 0.0),
                    )
                    or None,
                    "request_tags": request_tags_json,
                    "metadata": task_metadata_json,
                    "next_check_at": now
                    + timedelta(seconds=VOLCENGINE_VIDEO_POLL_INTERVAL_SECONDS),
                    "last_checked_at": now,
                    "last_error": None,
                },
            },
        )

    async def _poll_single_task(self, task: Any) -> None:
        credentials = self._get_volcengine_credentials(task=task)
        if credentials is None:
            raise ValueError(
                f"Could not resolve deployment credentials for model={task.model} model_id={task.model_id}"
            )

        provider = getattr(task, "custom_llm_provider", "volcengine") or "volcengine"
        if provider == "byteplus":
            from litellm.llms.byteplus.videos.transformation import (
                BytePlusVideoConfig,
            )

            config = BytePlusVideoConfig()
        else:
            config = VolcEngineVideoConfig()
        provider_model = task.provider_model or task.model or task.model_group or ""
        litellm_params = GenericLiteLLMParams(**credentials)
        headers = config.validate_environment(
            headers={},
            model=provider_model,
            api_key=credentials.get("api_key"),
            litellm_params=litellm_params,
        )
        api_base = config.get_complete_url(
            model=provider_model,
            api_base=credentials.get("api_base"),
            litellm_params=litellm_params.model_dump(exclude_none=True),
        )
        status_url, params = config.transform_video_status_retrieve_request(
            video_id=task.video_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        async_httpx_client = get_async_httpx_client(
            llm_provider=(
                LlmProviders.BYTEPLUS
                if provider == "byteplus"
                else LlmProviders.VOLCENGINE
            )
        )
        response = await async_httpx_client.client.get(
            status_url,
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        video_response = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=None,
            custom_llm_provider=provider,
        )
        await self._reconcile_task_from_video_response(
            video_id=task.video_id,
            video_response=video_response,
        )

    async def _reconcile_task_from_video_response(
        self,
        video_id: str,
        video_response: VideoObject,
        kwargs: Optional[dict] = None,
    ) -> None:
        video_task_table = self._get_video_task_table_model()
        if video_task_table is None:
            return

        if not video_id:
            return

        task = await video_task_table.find_unique(where={"video_id": video_id})
        if task is None:
            if kwargs is not None:
                await self._register_pending_video_task(
                    kwargs=kwargs,
                    completion_response=video_response,
                )
                task = await video_task_table.find_unique(where={"video_id": video_id})

        if task is None:
            verbose_proxy_logger.warning(
                "Volcengine video billing: task row missing for video_id=%s",
                video_id,
            )
            return

        provider_status = video_response.status or "queued"
        terminal_completed = provider_status == VOLCENGINE_VIDEO_COMPLETED_STATUS
        terminal_no_charge = provider_status in VOLCENGINE_VIDEO_NO_CHARGE_STATUSES
        now = _now_utc()

        if terminal_completed:
            await self._finalize_completed_task(
                task=task, video_response=video_response
            )
            return

        if terminal_no_charge:
            await video_task_table.update(
                where={"video_id": video_id},
                data={
                    "provider_status": provider_status,
                    "billing_state": "no_charge",
                    "completed_at": _ts_to_datetime(video_response.completed_at) or now,
                    "last_checked_at": now,
                    "next_check_at": None,
                    "last_error": None,
                },
            )
            return

        usage = dict(video_response.usage or {})
        await video_task_table.update(
            where={"video_id": video_id},
            data={
                "provider_status": provider_status,
                "duration_seconds": _safe_float(
                    usage.get("duration_seconds"),
                    default=_safe_float(video_response.seconds, 0.0),
                )
                or None,
                "last_checked_at": now,
                "next_check_at": now
                + timedelta(seconds=VOLCENGINE_VIDEO_POLL_INTERVAL_SECONDS),
                "check_attempts": {"increment": 1},
                "last_error": None,
            },
        )

    async def _finalize_completed_task(
        self,
        task: Any,
        video_response: VideoObject,
    ) -> None:
        video_task_table = self._get_video_task_table_model()
        if video_task_table is None:
            return

        if getattr(task, "billing_state", None) in {"billed", "no_charge", "settling"}:
            return

        claim_result = await video_task_table.update_many(
            where={"video_id": task.video_id, "billing_state": "pending"},
            data={
                "billing_state": "settling",
                "provider_status": video_response.status
                or VOLCENGINE_VIDEO_COMPLETED_STATUS,
                "last_checked_at": _now_utc(),
                "next_check_at": None,
                "last_error": None,
            },
        )
        if claim_result == 0:
            return

        now = _now_utc()
        usage = dict(video_response.usage or {})
        total_tokens = _safe_int(
            usage.get("total_tokens"),
            default=_safe_int(usage.get("completion_tokens")),
        )
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(
            usage.get("completion_tokens"),
            default=total_tokens,
        )
        duration_seconds = _safe_float(
            usage.get("duration_seconds"),
            default=_safe_float(video_response.seconds),
        )

        unit_price = self._resolve_final_unit_price(
            task=task, video_response=video_response
        )
        provider_final_spend = unit_price * float(total_tokens) / 1_000_000.0
        final_spend_usd_before_discount = _convert_provider_spend_to_usd(
            amount=provider_final_spend,
            currency=task.pricing_currency or "USD",
        )
        discount_factor = self._resolve_discount_factor(task=task)
        final_spend_usd = self._apply_discount_factor(
            spend=final_spend_usd_before_discount,
            discount_factor=discount_factor,
            task=task,
        )
        delta_spend = max(final_spend_usd - float(task.spend or 0.0), 0.0)
        provider_delta_spend = provider_final_spend
        delta_prompt_tokens = max(prompt_tokens - int(task.prompt_tokens or 0), 0)
        delta_completion_tokens = max(
            completion_tokens - int(task.completion_tokens or 0), 0
        )

        try:
            if (
                delta_spend > 0
                or delta_prompt_tokens > 0
                or delta_completion_tokens > 0
            ):
                await self._apply_async_billing_delta(
                    task=task,
                    delta_spend=delta_spend,
                    provider_delta_spend=provider_delta_spend,
                    delta_prompt_tokens=delta_prompt_tokens,
                    delta_completion_tokens=delta_completion_tokens,
                    usage=usage,
                    custom_discount_factor=discount_factor,
                )

                await self._upsert_final_spend_log(
                    task=task,
                    final_spend=final_spend_usd,
                    provider_spend_amount=provider_final_spend,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    usage=usage,
                    custom_discount_factor=discount_factor,
                )

            await video_task_table.update(
                where={"video_id": task.video_id},
                data={
                    "provider_status": video_response.status
                    or VOLCENGINE_VIDEO_COMPLETED_STATUS,
                    "billing_state": "billed",
                    "spend": final_spend_usd,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "duration_seconds": duration_seconds or None,
                    "completed_at": _ts_to_datetime(video_response.completed_at) or now,
                    "billed_at": now,
                    "last_checked_at": now,
                    "next_check_at": None,
                    "check_attempts": {"increment": 1},
                    "last_error": None,
                },
            )
        except Exception as e:
            await video_task_table.update(
                where={"video_id": task.video_id},
                data={
                    "billing_state": "pending",
                    "last_error": str(e),
                    "last_checked_at": now,
                    "next_check_at": now
                    + timedelta(seconds=VOLCENGINE_VIDEO_RETRY_INTERVAL_SECONDS),
                    "check_attempts": {"increment": 1},
                },
            )
            raise

    async def _apply_async_billing_delta(
        self,
        task: Any,
        delta_spend: float,
        provider_delta_spend: float,
        delta_prompt_tokens: int,
        delta_completion_tokens: int,
        usage: Dict[str, Any],
        custom_discount_factor: Optional[float] = None,
    ) -> None:
        from litellm.proxy.proxy_server import (
            litellm_proxy_budget_name,
            update_cache,
            user_api_key_cache,
        )

        spend_log_identity = await self._get_spend_log_identity(video_id=task.video_id)
        effective_api_key = spend_log_identity.get("api_key") or task.api_key or ""
        effective_user = spend_log_identity.get("user") or task.user or None
        effective_team_id = spend_log_identity.get("team_id") or task.team_id or None
        effective_org_id = (
            spend_log_identity.get("organization_id") or task.organization_id or None
        )
        effective_end_user = spend_log_identity.get("end_user") or task.end_user or None
        request_tags = _parse_request_tags(task.request_tags)
        delta_metadata_dict: Dict[str, Any] = {
            "usage_object": usage,
            "async_billing_only": True,
            "provider_spend_currency": task.pricing_currency or "CNY",
            "provider_spend_amount": provider_delta_spend,
            "billing_spend_currency": "USD",
            "billing_spend_amount": delta_spend,
            "provider_to_usd_fx_rate": (
                _get_cny_per_usd_rate()
                if (task.pricing_currency or "").upper() == "CNY"
                else None
            ),
            "video_billing_task_id": task.video_id,
        }
        if custom_discount_factor is not None:
            delta_metadata_dict["custom_discount_factor"] = custom_discount_factor
        delta_metadata: SpendLogsMetadata = cast(SpendLogsMetadata, delta_metadata_dict)
        delta_payload: SpendLogsPayload = cast(
            SpendLogsPayload,
            {
                "request_id": task.video_id,
                "call_type": "avideo_generation",
                "api_key": effective_api_key,
                "spend": delta_spend,
                "total_tokens": delta_prompt_tokens + delta_completion_tokens,
                "prompt_tokens": delta_prompt_tokens,
                "completion_tokens": delta_completion_tokens,
                "startTime": task.created_at,
                "endTime": task.created_at,
                "completionStartTime": None,
                "model": task.model or "",
                "model_id": task.model_id or "",
                "model_group": task.model_group or task.model or "",
                "mcp_namespaced_tool_name": None,
                "agent_id": None,
                "api_base": "",
                "user": effective_user or "",
                "metadata": json.dumps(delta_metadata),
                "cache_hit": "False",
                "cache_key": "Cache OFF",
                "request_tags": json.dumps(request_tags),
                "team_id": effective_team_id,
                "organization_id": effective_org_id,
                "end_user": effective_end_user,
                "requester_ip_address": None,
                "custom_llm_provider": task.custom_llm_provider or "volcengine",
                "messages": None,
                "response": None,
                "proxy_server_request": None,
                "session_id": None,
                "request_duration_ms": None,
                "status": "success",
            },
        )

        await self.db_spend_update_writer.apply_async_billing_delta(
            response_cost=delta_spend,
            user_id=effective_user,
            hashed_token=effective_api_key or None,
            team_id=effective_team_id,
            org_id=effective_org_id,
            end_user_id=effective_end_user,
            prisma_client=self.prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_proxy_budget_name=litellm_proxy_budget_name,
            payload=delta_payload,
            request_tags=request_tags,
        )

        await update_cache(
            token=effective_api_key or None,
            user_id=effective_user,
            end_user_id=effective_end_user,
            team_id=effective_team_id,
            response_cost=delta_spend,
            parent_otel_span=None,
            tags=request_tags,
        )

    async def _upsert_final_spend_log(
        self,
        task: Any,
        final_spend: float,
        provider_spend_amount: float,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        usage: Dict[str, Any],
        custom_discount_factor: Optional[float] = None,
    ) -> None:
        existing_spend_log = await self.prisma_client.db.litellm_spendlogs.find_unique(
            where={"request_id": task.video_id}
        )
        metadata = self._build_final_spend_log_metadata(
            existing_metadata=getattr(existing_spend_log, "metadata", None),
            usage=usage,
            pricing_currency=task.pricing_currency or "CNY",
            provider_spend_amount=provider_spend_amount,
            final_spend=final_spend,
            video_task_id=task.video_id,
            custom_discount_factor=custom_discount_factor,
        )
        metadata_json = _to_prisma_json(metadata)
        request_tags_json = _to_prisma_json(_parse_request_tags(task.request_tags))
        empty_json_object = _to_prisma_json({})

        if existing_spend_log is None:
            create_data = {
                "request_id": task.video_id,
                "call_type": "avideo_generation",
                "api_key": task.api_key or "",
                "spend": final_spend,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "startTime": task.created_at,
                "endTime": task.created_at,
                "model": task.model or "",
                "model_id": task.model_id or "",
                "model_group": task.model_group or task.model or "",
                "custom_llm_provider": task.custom_llm_provider or "volcengine",
                "api_base": "",
                "user": task.user or "",
                "metadata": metadata_json,
                "cache_hit": "False",
                "cache_key": "Cache OFF",
                "request_tags": request_tags_json,
                "team_id": task.team_id,
                "organization_id": task.organization_id,
                "end_user": task.end_user,
                "messages": empty_json_object,
                "response": empty_json_object,
                "proxy_server_request": empty_json_object,
                "status": "success",
            }
            await self.prisma_client.db.litellm_spendlogs.upsert(
                where={"request_id": task.video_id},
                data={
                    "create": create_data,
                    "update": {
                        "spend": final_spend,
                        "total_tokens": total_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "metadata": metadata_json,
                    },
                },
            )
            return

        await self.prisma_client.db.litellm_spendlogs.update(
            where={"request_id": task.video_id},
            data={
                "spend": final_spend,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "metadata": metadata_json,
            },
        )

    def _build_final_spend_log_metadata(
        self,
        existing_metadata: Any,
        usage: Dict[str, Any],
        pricing_currency: str,
        provider_spend_amount: float,
        final_spend: float,
        video_task_id: str,
        custom_discount_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        if isinstance(existing_metadata, dict):
            metadata_dict = dict(existing_metadata)
        else:
            metadata_dict = safe_json_loads(existing_metadata, default={})
            if not isinstance(metadata_dict, dict):
                metadata_dict = {}
        metadata_dict["usage_object"] = usage
        metadata_dict["provider_spend_currency"] = pricing_currency
        metadata_dict["provider_spend_amount"] = provider_spend_amount
        metadata_dict["billing_spend_currency"] = "USD"
        metadata_dict["billing_spend_amount"] = final_spend
        if pricing_currency.upper() == "CNY":
            metadata_dict["provider_to_usd_fx_rate"] = _get_cny_per_usd_rate()
        metadata_dict["video_billing_task_id"] = video_task_id
        if custom_discount_factor is not None:
            metadata_dict["custom_discount_factor"] = custom_discount_factor
        return metadata_dict

    async def _get_request_content_for_task_registration(
        self,
        kwargs: dict,
        completion_response: VideoObject,
    ) -> Optional[List[Dict[str, Any]]]:
        request_content = cast(
            Optional[List[Dict[str, Any]]],
            (getattr(completion_response, "_hidden_params", {}) or {}).get(
                "request_content"
            ),
        )
        if request_content is not None:
            return request_content

        request_content = self._build_request_content_from_proxy_server_request(
            kwargs.get("proxy_server_request")
        )
        if request_content is not None:
            return request_content

        return await self._get_request_content_from_existing_spend_log(
            video_id=completion_response.id or ""
        )

    async def _get_request_content_from_existing_spend_log(
        self,
        video_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        existing_spend_log = await self._get_existing_spend_log(video_id=video_id)
        if existing_spend_log is None:
            return None

        return self._build_request_content_from_proxy_server_request(
            getattr(existing_spend_log, "proxy_server_request", None)
        )

    async def _get_existing_spend_log(self, video_id: str) -> Optional[Any]:
        if not video_id or self.prisma_client is None:
            return None

        return await self.prisma_client.db.litellm_spendlogs.find_unique(
            where={"request_id": video_id}
        )

    async def _get_spend_log_identity(self, video_id: str) -> Dict[str, Optional[str]]:
        existing_spend_log = await self._get_existing_spend_log(video_id=video_id)
        if existing_spend_log is None:
            return {
                "api_key": None,
                "user": None,
                "team_id": None,
                "organization_id": None,
                "end_user": None,
            }

        return {
            "api_key": _normalize_non_empty_str(
                getattr(existing_spend_log, "api_key", None)
            ),
            "user": _normalize_non_empty_str(getattr(existing_spend_log, "user", None)),
            "team_id": _normalize_non_empty_str(
                getattr(existing_spend_log, "team_id", None)
            ),
            "organization_id": _normalize_non_empty_str(
                getattr(existing_spend_log, "organization_id", None)
            ),
            "end_user": _normalize_non_empty_str(
                getattr(existing_spend_log, "end_user", None)
            ),
        }

    def _build_request_content_from_proxy_server_request(
        self,
        proxy_server_request: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        if isinstance(proxy_server_request, str):
            proxy_server_request = safe_json_loads(proxy_server_request, default={})
        if not isinstance(proxy_server_request, dict):
            return None

        prompt = proxy_server_request.get("prompt") or ""
        provided_content = proxy_server_request.get("content")
        input_reference = proxy_server_request.get("input_reference")

        try:
            return VolcEngineVideoConfig()._build_content_list(
                prompt=str(prompt),
                provided_content=deepcopy(provided_content),
                input_reference=deepcopy(input_reference),
            )
        except Exception:
            return None

    def _get_api_key_hash(
        self,
        metadata: dict,
        standard_logging_object: Optional[StandardLoggingPayload],
    ) -> str:
        if standard_logging_object is not None:
            api_key_hash = (standard_logging_object.get("metadata", {}) or {}).get(
                "user_api_key_hash"
            ) or ""
            if api_key_hash:
                return str(api_key_hash)
        return _hash_token_if_needed(cast(Optional[str], metadata.get("user_api_key")))

    def _get_request_tags(
        self,
        metadata: dict,
        standard_logging_object: Optional[StandardLoggingPayload],
    ) -> List[str]:
        if (
            standard_logging_object is not None
            and standard_logging_object.get("request_tags") is not None
        ):
            return _parse_request_tags(standard_logging_object.get("request_tags"))
        return _parse_request_tags(metadata.get("tags"))

    def _snapshot_custom_discount(self, metadata: dict) -> Optional[Dict[str, Any]]:
        """
        Snapshot the team custom discount map at task registration so the billed
        rate is locked in even if the discount is changed before the async video
        task completes (completion can lag the request by minutes to hours).

        Team discounts surface either directly on metadata.custom_discount or
        nested under metadata.user_api_key_auth_metadata.custom_discount (the
        proxy mirrors team metadata there), so both locations are checked to stay
        consistent with the synchronous chat billing hook.
        """
        custom_discount = metadata.get("custom_discount")
        if not isinstance(custom_discount, dict) or not custom_discount:
            auth_metadata = metadata.get("user_api_key_auth_metadata")
            if isinstance(auth_metadata, dict):
                custom_discount = auth_metadata.get("custom_discount")
        if not isinstance(custom_discount, dict) or not custom_discount:
            return None
        return deepcopy(custom_discount)

    def _resolve_discount_factor(self, task: Any) -> Optional[float]:
        task_metadata = self._parse_task_metadata(task)
        custom_discount = task_metadata.get("custom_discount")
        if not isinstance(custom_discount, dict) or not custom_discount:
            return None
        return self._match_discount_factor(
            custom_discount=custom_discount,
            model=task.model or "",
            model_group=task.model_group or "",
            provider_model=task.provider_model or "",
        )

    def _apply_discount_factor(
        self,
        spend: float,
        discount_factor: Optional[float],
        task: Any,
    ) -> float:
        """
        Apply the discount as a multiplier (final = spend * factor), matching the
        chat billing semantics where erp stores the factor in (0, 1] and 0.8
        means an 80% charge, not a 20% charge.
        """
        if discount_factor is None or spend <= 0:
            return spend
        discounted_spend = spend * discount_factor
        verbose_proxy_logger.info(
            "Volcengine video billing applied discount: video_id=%s model=%s "
            "discount_factor=%s original_spend_usd=%.6f discounted_spend_usd=%.6f",
            getattr(task, "video_id", None),
            task.model or "",
            discount_factor,
            spend,
            discounted_spend,
        )
        return discounted_spend

    def _match_discount_factor(
        self,
        custom_discount: Dict[str, Any],
        model: str,
        model_group: str,
        provider_model: str,
    ) -> Optional[float]:
        """
        Match a discount factor for the task using the same precedence as the
        chat billing hook: exact request model, provider/model short name,
        model_group, provider model, then a lenient substring fallback.
        """
        exact_candidates: List[str] = []
        if model:
            exact_candidates.append(model)
            if "/" in model:
                exact_candidates.append(model.rsplit("/", 1)[-1])
        if model_group:
            exact_candidates.append(model_group)
        if provider_model:
            exact_candidates.append(provider_model)

        for candidate in exact_candidates:
            if candidate and candidate in custom_discount:
                factor = self._normalize_discount_factor(custom_discount[candidate])
                if factor is not None:
                    return factor

        for key, value in custom_discount.items():
            if not key or not model:
                continue
            if key in model or model in key:
                factor = self._normalize_discount_factor(value)
                if factor is not None:
                    return factor
        return None

    @staticmethod
    def _normalize_discount_factor(value: Any) -> Optional[float]:
        factor = _safe_float(value, default=0.0)
        if factor <= 0.0 or factor > 1.0:
            return None
        return factor

    def _parse_task_metadata(self, task: Any) -> Dict[str, Any]:
        metadata = getattr(task, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            parsed = safe_json_loads(metadata, default={})
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _resolve_pricing_model(self, model_info: dict) -> str:
        pricing_model = (
            model_info.get("provider_pricing_model")
            or model_info.get("base_model")
            or VOLCENGINE_VIDEO_DEFAULT_PRICING_MODEL
        )
        normalized_model = _normalize_pricing_model(str(pricing_model))
        if not (
            model_info.get("provider_pricing_model") or model_info.get("base_model")
        ):
            verbose_proxy_logger.warning(
                "Ark video billing falling back to default pricing model=%s. "
                "Set model_info.provider_pricing_model or model_info.base_model for exact endpoint pricing.",
                normalized_model,
            )
        return normalized_model

    def _resolve_final_unit_price(
        self,
        task: Any,
        video_response: VideoObject,
    ) -> float:
        """
        Resolution-dependent pricing (e.g. byteplus/dreamina-seedance-2.0 charges
        more for 1080p) means the registration-time snapshot can be wrong when the
        request did not pin a resolution. The completed task reports the actual
        output resolution, so re-resolve against it and fall back to the snapshot
        when the provider omits it or pricing lookup fails.
        """
        resolution = self._extract_response_resolution(video_response)
        if resolution is not None and task.pricing_model:
            generate_audio = self._extract_response_generate_audio(task, video_response)
            try:
                unit_price, _ = self._resolve_pricing_snapshot(
                    pricing_model=task.pricing_model,
                    has_input_video=bool(task.has_input_video),
                    resolution=resolution,
                    generate_audio=generate_audio,
                )
                return unit_price
            except Exception as e:
                verbose_proxy_logger.warning(
                    "Ark video billing could not re-resolve price for video_id=%s "
                    "resolution=%s: %s. Falling back to snapshot price.",
                    task.video_id,
                    resolution,
                    str(e),
                )
        return float(task.price_per_million_tokens or 0.0)

    @staticmethod
    def _extract_response_resolution(video_response: VideoObject) -> Optional[str]:
        hidden_params = getattr(video_response, "_hidden_params", {}) or {}
        resolution = hidden_params.get("resolution")
        if resolution:
            return str(resolution)
        usage = video_response.usage or {}
        usage_resolution = usage.get("resolution") or usage.get("video_resolution")
        return str(usage_resolution) if usage_resolution else None

    def _extract_request_resolution(
        self,
        kwargs: dict,
        completion_response: VideoObject,
    ) -> Optional[str]:
        optional_params = kwargs.get("optional_params")
        if isinstance(optional_params, dict):
            resolution = optional_params.get("resolution") or optional_params.get(
                "size"
            )
            if resolution:
                return str(resolution)

        proxy_server_request = kwargs.get("proxy_server_request")
        if isinstance(proxy_server_request, str):
            proxy_server_request = safe_json_loads(proxy_server_request, default={})
        if isinstance(proxy_server_request, dict):
            resolution = proxy_server_request.get(
                "resolution"
            ) or proxy_server_request.get("size")
            if resolution:
                return str(resolution)

        return self._extract_response_resolution(completion_response)

    def _extract_request_generate_audio(
        self,
        kwargs: dict,
        completion_response: VideoObject,
    ) -> bool:
        optional_params = kwargs.get("optional_params")
        if isinstance(optional_params, dict) and "generate_audio" in optional_params:
            return _coerce_bool(optional_params.get("generate_audio"))

        proxy_server_request = kwargs.get("proxy_server_request")
        if isinstance(proxy_server_request, str):
            proxy_server_request = safe_json_loads(proxy_server_request, default={})
        if (
            isinstance(proxy_server_request, dict)
            and "generate_audio" in proxy_server_request
        ):
            return _coerce_bool(proxy_server_request.get("generate_audio"))

        hidden_params = getattr(completion_response, "_hidden_params", {}) or {}
        return _coerce_bool(hidden_params.get("generate_audio"))

    def _extract_response_generate_audio(
        self,
        task: Any,
        video_response: VideoObject,
    ) -> bool:
        hidden_params = getattr(video_response, "_hidden_params", {}) or {}
        if "generate_audio" in hidden_params:
            return _coerce_bool(hidden_params.get("generate_audio"))
        return _coerce_bool(self._parse_task_metadata(task).get("generate_audio"))

    def _resolve_pricing_snapshot(
        self,
        pricing_model: str,
        has_input_video: bool,
        resolution: Any = None,
        generate_audio: bool = False,
    ) -> Tuple[float, str]:
        self._ensure_runtime_pricing_models_registered()

        audio_base_keys = (
            "volcengine_video_output_cost_per_million_tokens_without_audio",
            "volcengine_video_output_cost_per_million_tokens_with_audio",
        )
        input_video_base_keys = (
            "volcengine_video_output_cost_per_million_tokens_without_input_video",
            "volcengine_video_output_cost_per_million_tokens_with_input_video",
        )

        # The proxy registers each deployment's litellm_params.model (e.g.
        # "byteplus/dreamina-seedance-2-0-260128") into litellm.model_cost without
        # the Volcengine video pricing keys, so only accept a candidate whose entry
        # actually carries pricing; otherwise fall through to the dotted/versionless
        # variant (e.g. "byteplus/dreamina-seedance-2.0") that does.
        pricing_entry = None
        pricing_key = None
        for candidate in _candidate_pricing_models(pricing_model):
            entry = litellm.model_cost.get(candidate)
            if entry is None:
                continue
            if any(
                entry.get(key) is not None
                for key in audio_base_keys + input_video_base_keys
            ):
                pricing_entry = entry
                pricing_key = candidate
                break

        if pricing_entry is None or pricing_key is None:
            raise ValueError(
                f"No Volcengine video pricing config found for model={pricing_model}"
            )

        # Seedance 1.5 Pro prices by whether audio was generated; the 2.0 family
        # prices by whether the request referenced an input video. Use whichever
        # dimension the matched entry actually defines.
        if any(pricing_entry.get(key) is not None for key in audio_base_keys):
            base_price_key = (
                "volcengine_video_output_cost_per_million_tokens_with_audio"
                if generate_audio
                else "volcengine_video_output_cost_per_million_tokens_without_audio"
            )
        else:
            base_price_key = (
                "volcengine_video_output_cost_per_million_tokens_with_input_video"
                if has_input_video
                else "volcengine_video_output_cost_per_million_tokens_without_input_video"
            )

        price_key = base_price_key
        if _is_1080p_resolution(resolution):
            resolution_price_key = f"{base_price_key}_1080p"
            if pricing_entry.get(resolution_price_key) is not None:
                price_key = resolution_price_key

        unit_price = pricing_entry.get(price_key)
        if unit_price is None:
            raise ValueError(
                f"Missing pricing key={price_key} for Volcengine video model={pricing_key}"
            )
        pricing_currency = pricing_entry.get("provider_pricing_currency", "CNY")
        return float(unit_price), str(pricing_currency)

    def _ensure_runtime_pricing_models_registered(self) -> None:
        if self._pricing_models_registered:
            return
        register_ark_video_pricing_models()
        self._pricing_models_registered = True

    def _get_video_task_table_model(self) -> Optional[Any]:
        if self._video_task_table_unavailable or self.prisma_client is None:
            return None

        try:
            return getattr(self.prisma_client.db, "litellm_videotasktable")
        except AttributeError:
            self._video_task_table_unavailable = True
            verbose_proxy_logger.warning(
                "Volcengine video billing disabled: Prisma client is missing "
                "litellm_videotasktable. Run `poetry run prisma generate`, apply the "
                "latest proxy migrations, and restart the proxy."
            )
            return None

    def _get_volcengine_credentials(self, task: Any) -> Optional[Dict[str, Any]]:
        for candidate in (
            task.model_id,
            task.model_group,
            task.model,
            task.provider_model,
        ):
            if not candidate:
                continue
            credentials = self.llm_router.get_deployment_credentials_with_provider(
                candidate
            )
            if (
                credentials
                and credentials.get("custom_llm_provider") in ARK_VIDEO_PROVIDERS
            ):
                return credentials
        return None
