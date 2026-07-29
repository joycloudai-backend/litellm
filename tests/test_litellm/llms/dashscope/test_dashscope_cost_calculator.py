"""
Test suite for Dashscope cost calculation functionality.

DashScope bills tiered models with step pricing: the request's total input
tokens select one tier and the entire request is charged at that tier's
rates. These tests validate:
- Flat-rate fallback for non-tiered models.
- Tier selection by total input tokens (including the tier upper boundary).
- All token types (text, cached, completion, reasoning) billed at the
  selected tier, never split across tiers.
- Requests beyond the last tier's range billed at the last tier.
"""

import math
import os
import sys

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.dashscope.cost_calculator import (
    cost_per_token as dashscope_cost_per_token,
)
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


class TestDashscopeCostCalculator:
    """Test suite for Dashscope cost calculation functionality."""

    @pytest.fixture(autouse=True)
    def setup_model_cost_map(self):
        """Set up the model cost map for testing by loading it locally."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_dashscope_flat_pricing_fallback(self):
        """
        Tests that the dashscope calculator falls back to flat pricing for models
        without a 'tiered_pricing' key (e.g., qwen-max).
        """
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-max", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-max")
        expected_prompt_cost = 1000 * model_info["input_cost_per_token"]
        expected_completion_cost = 500 * model_info["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_within_first_tier(self):
        """
        Tests step pricing when the input token count falls in the first tier.
        Uses 'dashscope/qwen-flash' as a real-world example.
        """
        # Tier 1 for qwen-flash is [0, 256,000] tokens
        usage = Usage(prompt_tokens=100000, completion_tokens=50000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1_pricing = model_info["tiered_pricing"][0]

        expected_prompt_cost = 100000 * tier_1_pricing["input_cost_per_token"]
        expected_completion_cost = 50000 * tier_1_pricing["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_at_tier_boundary_uses_lower_tier(self):
        """
        DashScope ranges are '0 < Token <= 256K', so exactly 256,000 input
        tokens must still be billed at tier 1.
        """
        usage = Usage(prompt_tokens=256000, completion_tokens=1000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]

        assert math.isclose(
            prompt_cost, 256000 * tier_1["input_cost_per_token"], rel_tol=1e-10
        )
        assert math.isclose(
            completion_cost, 1000 * tier_1["output_cost_per_token"], rel_tol=1e-10
        )

    def test_dashscope_tiered_pricing_step_billing_second_tier(self):
        """
        Step pricing: 300k input tokens exceed tier 1's 256k upper bound, so the
        ENTIRE request (input and output) is billed at tier 2. A graduated
        (segment-summed) calculation would produce a lower, incorrect cost.
        """
        # Tiering for qwen-flash: Tier 1: [0, 256k], Tier 2: [256k, 1M]
        usage = Usage(prompt_tokens=300000, completion_tokens=300000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = 300000 * tier_2["input_cost_per_token"]
        expected_completion_cost = 300000 * tier_2["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)
        # Guard against a regression to graduated billing
        graduated_prompt_cost = (256000 * tier_1["input_cost_per_token"]) + (
            44000 * tier_2["input_cost_per_token"]
        )
        assert prompt_cost > graduated_prompt_cost

    def test_dashscope_tiered_pricing_with_caching(self):
        """
        Tier selection uses TOTAL input tokens (cached + new); cached tokens are
        then billed at the selected tier's cache-read rate.
        Uses qwen3-coder-plus, which has cache-specific pricing per tier.
        """
        usage = Usage(
            prompt_tokens=50000,  # 10k cached + 40k new
            completion_tokens=1000,
            total_tokens=51000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=10000),
        )

        prompt_cost, _ = dashscope_cost_per_token(model="qwen3-coder-plus", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen3-coder-plus")
        # qwen3-coder-plus tiers: [0, 32k], [32k, 128k], ... → 50k total input → tier 2
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = (10000 * tier_2["cache_read_input_token_cost"]) + (
            40000 * tier_2["input_cost_per_token"]
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_exceeding_highest_tier(self):
        """
        Input tokens beyond the last tier's upper bound are billed entirely at
        the last tier's rate.
        """
        usage = Usage(
            prompt_tokens=1200000, completion_tokens=1000
        )  # Max defined range for qwen-flash is 1M

        prompt_cost, _ = dashscope_cost_per_token(model="qwen-flash", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        last_tier = model_info["tiered_pricing"][-1]

        expected_prompt_cost = 1200000 * last_tier["input_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
