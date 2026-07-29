"""
Cost calculator for Dashscope Chat models.

DashScope tiered models use step pricing: the request's total input tokens
select a single price tier, and the whole request (input, cached input,
output, reasoning) is billed at that tier's rates. Tiers are never split
and summed segment by segment.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


@dataclass
class TokenBreakdown:
    """Token breakdown for cost calculation."""

    text_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int


def _extract_token_breakdown(usage: Usage) -> TokenBreakdown:
    """Extract token counts from usage, handling cached and reasoning tokens."""
    cached_tokens = 0
    if usage.prompt_tokens_details and hasattr(
        usage.prompt_tokens_details, "cached_tokens"
    ):
        cached_tokens = usage.prompt_tokens_details.cached_tokens or 0

    text_tokens = usage.prompt_tokens - cached_tokens

    reasoning_tokens = 0
    if (
        hasattr(usage, "completion_tokens_details")
        and usage.completion_tokens_details
        and hasattr(usage.completion_tokens_details, "reasoning_tokens")
    ):
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0

    completion_tokens = (usage.completion_tokens or 0) - reasoning_tokens

    return TokenBreakdown(
        text_tokens, cached_tokens, completion_tokens, reasoning_tokens
    )


def _select_tier(prompt_tokens: int, tiered_pricing: List[dict]) -> dict:
    """Pick the single tier whose input-token range covers the request.

    DashScope ranges are half-open on the left: ``[0, 256000]`` means
    ``0 < tokens <= 256000``. Requests beyond the last tier's upper bound are
    billed entirely at the last tier's rates.
    """
    sorted_tiers = sorted(tiered_pricing, key=lambda t: t.get("range", [0, 0])[0])
    for tier in sorted_tiers:
        tier_range = tier.get("range", [])
        if len(tier_range) == 2 and prompt_tokens <= tier_range[1]:
            return tier
    return sorted_tiers[-1]


def _tier_cost(tier: dict, cost_key: str, fallback_cost_key: Optional[str]) -> float:
    value = tier.get(cost_key)
    if value is None and fallback_cost_key is not None:
        value = tier.get(fallback_cost_key)
    return float(value or 0.0)


def _calculate_prompt_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tier: Optional[dict],
) -> float:
    """Calculate total prompt cost including cached tokens."""
    if tier is not None:
        input_cost = _tier_cost(tier, "input_cost_per_token", None)
        cache_cost = _tier_cost(
            tier, "cache_read_input_token_cost", "input_cost_per_token"
        )
    else:
        input_cost = float(model_info.get("input_cost_per_token") or 0.0)
        cache_cost_val = model_info.get("cache_read_input_token_cost")
        cache_cost = input_cost if cache_cost_val is None else float(cache_cost_val)

    return (breakdown.text_tokens * input_cost) + (breakdown.cached_tokens * cache_cost)


def _calculate_completion_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tier: Optional[dict],
) -> float:
    """Calculate total completion cost including reasoning tokens."""
    if tier is not None:
        output_cost = _tier_cost(tier, "output_cost_per_token", None)
        reasoning_cost = _tier_cost(
            tier, "output_cost_per_reasoning_token", "output_cost_per_token"
        )
    else:
        output_cost = float(model_info.get("output_cost_per_token") or 0.0)
        reasoning_cost_val = model_info.get("output_cost_per_reasoning_token")
        reasoning_cost = (
            output_cost if reasoning_cost_val is None else float(reasoning_cost_val)
        )

    return (breakdown.completion_tokens * output_cost) + (
        breakdown.reasoning_tokens * reasoning_cost
    )


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculate cost per token for Dashscope models.

    Supports both step-tiered and flat pricing with cached and reasoning tokens.

    Args:
        model: Model name without provider prefix
        usage: LiteLLM Usage block

    Returns:
        Tuple[float, float] - (prompt_cost_in_usd, completion_cost_in_usd)
    """
    model_info = get_model_info(model=model, custom_llm_provider="dashscope")
    breakdown = _extract_token_breakdown(usage)
    tiered_pricing = model_info.get("tiered_pricing")
    tier = (
        _select_tier(usage.prompt_tokens or 0, tiered_pricing)
        if isinstance(tiered_pricing, list) and tiered_pricing
        else None
    )

    prompt_cost = _calculate_prompt_cost(
        breakdown=breakdown, model_info=model_info, tier=tier
    )
    completion_cost = _calculate_completion_cost(
        breakdown=breakdown, model_info=model_info, tier=tier
    )

    return prompt_cost, completion_cost
