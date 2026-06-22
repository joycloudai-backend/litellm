"""
Ensure /v1/model/info exposes Volcengine/BytePlus video pricing.

The video pricing keys live in code (not the static cost map) and are keyed by
dotted/versionless ids, so the model_info enrichment must resolve them via
provider_pricing_model / base_model and surface the custom
volcengine_video_output_cost_per_million_tokens_* fields for the dashboard.
"""

import litellm.proxy.proxy_server as ps


def _ark_video_deployment():
    return {
        "model_name": "seedance-2.0",
        "litellm_params": {
            "model": "volcengine/ep-m-20260612110901-2fmwx",
            "api_key": "sk-secret",
        },
        "model_info": {
            "mode": "video_generation",
            "base_model": "volcengine/doubao-seedance-2-0-260128",
            "provider_pricing_model": "volcengine/doubao-seedance-2.0",
        },
    }


def test_enrich_exposes_video_pricing_custom_keys():
    enriched = ps._enrich_model_info_with_litellm_data(model=_ark_video_deployment())
    model_info = enriched["model_info"]

    assert (
        model_info[
            "volcengine_video_output_cost_per_million_tokens_without_input_video"
        ]
        == 46.0
    )
    assert (
        model_info["volcengine_video_output_cost_per_million_tokens_with_input_video"]
        == 28.0
    )
    assert model_info["provider_pricing_currency"] == "CNY"
    # sensitive params must still be stripped by the enrichment
    assert "api_key" not in enriched["litellm_params"]


def test_enrich_skips_non_ark_models():
    deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"base_model": "gpt-4o"},
    }
    enriched = ps._enrich_model_info_with_litellm_data(model=deployment)
    assert (
        "volcengine_video_output_cost_per_million_tokens_without_input_video"
        not in enriched["model_info"]
    )
