import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import BudgetConfig


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_dict_does_not_require_litellm_params_instantiation(
    disable_budget_sync, monkeypatch
):
    class RaiseOnInit:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "LiteLLM_Params should not be instantiated in hot path"
            )

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.LiteLLM_Params",
        RaiseOnInit,
    )

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    deployment = {"litellm_params": {"model": "openai/gpt-4o-mini"}}
    provider = provider_budget._get_llm_provider_for_deployment(deployment)

    assert provider == "openai"


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_dict_view_supports_mapping_and_attr_access(
    disable_budget_sync, monkeypatch
):
    observed = {}

    def _future_style_get_llm_provider(
        model,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
        litellm_params=None,
    ):
        assert litellm_params is not None
        observed["model_attr"] = litellm_params.model
        observed["provider_get"] = litellm_params.get("custom_llm_provider")
        observed["api_base_item"] = litellm_params["api_base"]
        observed["has_api_key"] = "api_key" in litellm_params
        observed["model_dump"] = litellm_params.model_dump()
        return model, "openai", None, None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.litellm.get_llm_provider",
        _future_style_get_llm_provider,
    )

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    deployment = {
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "custom_llm_provider": "openai",
            "api_base": "https://api.openai.com/v1",
        }
    }
    provider = provider_budget._get_llm_provider_for_deployment(deployment)

    assert provider == "openai"
    assert observed["model_attr"] == "openai/gpt-4o-mini"
    assert observed["provider_get"] == "openai"
    assert observed["api_base_item"] == "https://api.openai.com/v1"
    assert observed["has_api_key"] is False
    assert observed["model_dump"]["model"] == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_async_filter_deployments_resolves_provider_once_per_deployment(
    disable_budget_sync, monkeypatch
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100.0),
        },
    )

    healthy_deployments = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    provider_resolution_calls = 0

    def _count_provider_calls(deployment):
        nonlocal provider_resolution_calls
        provider_resolution_calls += 1
        return "openai"

    monkeypatch.setattr(
        provider_budget,
        "_get_llm_provider_for_deployment",
        _count_provider_calls,
    )

    filtered_deployments = await provider_budget.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )

    assert len(filtered_deployments) == len(healthy_deployments)
    assert provider_resolution_calls == len(healthy_deployments)


@pytest.mark.asyncio
async def test_async_filter_deployments_does_not_recompute_provider_when_resolved_none(
    disable_budget_sync, monkeypatch
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100.0),
        },
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "max_budget": 100.0,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "deployment-1"},
            }
        ],
    )

    healthy_deployments = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "unknown-provider/model"},
            "model_info": {"id": "deployment-1"},
        }
    ]

    provider_resolution_calls = 0

    def _provider_returns_none(deployment):
        nonlocal provider_resolution_calls
        provider_resolution_calls += 1
        return None

    monkeypatch.setattr(
        provider_budget,
        "_get_llm_provider_for_deployment",
        _provider_returns_none,
    )

    filtered_deployments = await provider_budget.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )

    assert len(filtered_deployments) == len(healthy_deployments)
    assert provider_resolution_calls == len(healthy_deployments)


def _legacy_provider_resolution(deployment):
    """
    Reference implementation used before hot-path optimization.
    """
    try:
        _litellm_params = LiteLLM_Params(
            **deployment.get("litellm_params", {"model": ""})
        )
        _, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=_litellm_params.model,
            litellm_params=_litellm_params,
        )
    except Exception:
        return None
    return custom_llm_provider


@pytest.mark.parametrize(
    "deployment",
    [
        {"litellm_params": {"model": "openai/gpt-4o-mini"}},
        {"litellm_params": {"model": "gpt-4o-mini", "custom_llm_provider": "openai"}},
        {"litellm_params": {"model": "unknown-provider/model"}},
    ],
)
@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_matches_legacy_behavior(
    disable_budget_sync, deployment
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    current_provider = provider_budget._get_llm_provider_for_deployment(deployment)
    legacy_provider = _legacy_provider_resolution(deployment)

    assert current_provider == legacy_provider


def test_register_deployment_budget_for_runtime_added_deployment(
    disable_budget_sync, monkeypatch
):
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )
    model_id = "dynamic-deployment-id"
    budget_limiter.register_deployment_budget(
        deployment={
            "model_name": "dynamic-budget-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "max_budget": 0.000000000001,
                "budget_duration": "1d",
            },
            "model_info": {"id": model_id},
        }
    )

    config = budget_limiter._get_budget_config_for_deployment(model_id)
    assert config is not None
    assert config.max_budget == 0.000000000001
    assert config.budget_duration == "1d"

    budget_limiter.unregister_deployment_budget(model_id=model_id)
    assert budget_limiter._get_budget_config_for_deployment(model_id) is None


def _grouped_model_list(budget_group="acct-42", max_budget=10.0):
    return [
        {
            "model_name": "kimi-k2.5",
            "litellm_params": {
                "model": "openai/kimi-k2.5",
                "max_budget": max_budget,
                "budget_duration": "30d",
                "budget_group": budget_group,
            },
            "model_info": {"id": "deployment-kimi"},
        },
        {
            "model_name": "qwen-plus",
            "litellm_params": {
                "model": "openai/qwen-plus",
                "max_budget": max_budget,
                "budget_duration": "30d",
                "budget_group": budget_group,
            },
            "model_info": {"id": "deployment-qwen"},
        },
    ]


def test_budget_group_deployments_share_spend_key(disable_budget_sync, monkeypatch):
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config=None,
        model_list=_grouped_model_list(),
    )

    kimi_key = budget_limiter._deployment_spend_key("deployment-kimi", "30d")
    qwen_key = budget_limiter._deployment_spend_key("deployment-qwen", "30d")
    assert kimi_key == qwen_key == "deployment_spend:group:acct-42:30d"
    assert (
        budget_limiter._deployment_budget_start_time_key("deployment-kimi")
        == budget_limiter._deployment_budget_start_time_key("deployment-qwen")
        == "deployment_budget_start_time:group:acct-42"
    )

    ungrouped_key = budget_limiter._deployment_spend_key("other-deployment", "30d")
    assert ungrouped_key == "deployment_spend:other-deployment:30d"

    budget_limiter.unregister_deployment_budget(model_id="deployment-kimi")
    assert (
        budget_limiter._deployment_spend_key("deployment-kimi", "30d")
        == "deployment_spend:deployment-kimi:30d"
    )


@pytest.mark.asyncio
async def test_budget_group_spend_pooled_across_deployments(
    disable_budget_sync, monkeypatch
):
    """
    Spend on one deployment in a budget_group must count against every other
    deployment in the same group: 6 + 6 > 10 blocks both kimi and qwen.
    """
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config=None,
        model_list=_grouped_model_list(max_budget=10.0),
    )

    async def _log_spend(model_id, cost):
        await budget_limiter.async_log_success_event(
            kwargs={
                "standard_logging_object": {
                    "response_cost": cost,
                    "model_id": model_id,
                },
                "litellm_params": {"custom_llm_provider": "openai"},
            },
            response_obj=None,
            start_time=None,
            end_time=None,
        )

    healthy_deployments = _grouped_model_list(max_budget=10.0)

    await _log_spend("deployment-kimi", 6.0)
    filtered = await budget_limiter.async_filter_deployments(
        model="kimi-k2.5",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )
    assert len(filtered) == 2

    await _log_spend("deployment-qwen", 6.0)
    with pytest.raises(ValueError, match="Exceeded budget for deployment"):
        await budget_limiter.async_filter_deployments(
            model="qwen-plus",
            healthy_deployments=healthy_deployments,
            messages=[],
            request_kwargs={},
            parent_otel_span=None,
        )


@pytest.mark.asyncio
async def test_deployment_without_budget_group_keeps_isolated_spend(
    disable_budget_sync, monkeypatch
):
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)
    model_list = [
        {
            "model_name": "solo-model",
            "litellm_params": {
                "model": "openai/solo-model",
                "max_budget": 10.0,
                "budget_duration": "30d",
            },
            "model_info": {"id": "deployment-solo"},
        }
    ] + _grouped_model_list(max_budget=10.0)
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config=None,
        model_list=model_list,
    )

    await budget_limiter.dual_cache.async_set_cache(
        key="deployment_spend:group:acct-42:30d", value=100.0
    )

    filtered = await budget_limiter.async_filter_deployments(
        model="solo-model",
        healthy_deployments=model_list,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )
    assert [d["model_info"]["id"] for d in filtered] == ["deployment-solo"]


def test_router_add_deployment_registers_deployment_budget(
    disable_budget_sync, monkeypatch
):
    import asyncio

    from litellm import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)

    router = Router(
        model_list=[],
        optional_pre_call_checks=[],
    )

    router.add_deployment(
        deployment=Deployment(
            model_name="dynamic-budget-model",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4o-mini",
                api_key="fake-key",
                max_budget=0.000000000001,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="runtime-budget-deployment"),
        )
    )

    budget_limiter = router._get_router_deployment_budget_limiter()
    assert budget_limiter is not None
    config = budget_limiter._get_budget_config_for_deployment(
        "runtime-budget-deployment"
    )
    assert config is not None
    assert config.max_budget == 0.000000000001
