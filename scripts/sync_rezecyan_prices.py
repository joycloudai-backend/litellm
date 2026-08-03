#!/usr/bin/env python3
"""
Generate `rezecyan/*` entries in model_prices_and_context_window.json (and the
in-package backup JSON) from the CNY price sheet scripts/rezecyan_prices.json.

The FX rate lives ONLY in the source sheet (`cny_per_usd`); generated entries
carry audit fields (original CNY price, rate, as-of date) so any entry can be
reconciled against the vendor bill. Never hand-edit generated entries — edit
the source sheet and re-run:

    python3 scripts/sync_rezecyan_prices.py            # write both JSON files
    python3 scripts/sync_rezecyan_prices.py --dry-run  # print entries only
    python3 scripts/sync_rezecyan_prices.py --self-test
    python3 scripts/sync_rezecyan_prices.py --scrape   # Playwright: pricing page -> sheet -> price JSONs

本脚本默认抓取 default 分组价格（无需登录）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHEET = REPO_ROOT / "scripts" / "rezecyan_prices.json"
TARGET_FILES = [
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
]

PRICING_URL = "https://cn.rezecyan.com/pricing"
LOGIN_URL = "https://cn.rezecyan.com/login"
TARGET_GROUP = "default"
# default 分组无需登录即可获取；登录态文件保留供未来其他分组使用
STATE_FILE = Path.home() / ".cache" / "rezecyan_state.json"

# 人民币每百万 token 价 -> LiteLLM 每 token 美元价字段
CNY_PER_1M_FIELD_MAP = {
    "input_cny_per_1m": "input_cost_per_token",
    "output_cny_per_1m": "output_cost_per_token",
    "cache_read_cny_per_1m": "cache_read_input_token_cost",
    "cache_creation_cny_per_1m": "cache_creation_input_token_cost",
}
# 人民币按张计价 -> 每张美元价（图片模型）
CNY_PER_ITEM_FIELD_MAP = {
    "image_cny_per_image": "input_cost_per_image",
}

_PRICE_FIELDS = set(CNY_PER_1M_FIELD_MAP) | set(CNY_PER_ITEM_FIELD_MAP)

# billing_expr 系数字段 -> 源文件 CNY 字段；tier(...) 内为 USD/1M，页面再 × usd_exchange_rate
_TIER_FIELD_MAP = {
    "p": "input_cny_per_1m",
    "c": "output_cny_per_1m",
    "cr": "cache_read_cny_per_1m",
    "cc": "cache_creation_cny_per_1m",
}
_ABOVE_USD_FIELD = {
    "input_cny_per_1m": "input_cost_per_token",
    "output_cny_per_1m": "output_cost_per_token",
    "cache_read_cny_per_1m": "cache_read_input_token_cost",
    "cache_creation_cny_per_1m": "cache_creation_input_token_cost",
}
_TIER_RE = re.compile(
    r'(?:((?:(?:p|c|len)\s*(?:<|<=|>|>=)\s*[\d.eE+]+)'
    r'(?:\s*&&\s*(?:p|c|len)\s*(?:<|<=|>|>=)\s*[\d.eE+]+)*)\s*\?\s*)?'
    r'tier\("([^"]*)",\s*([^)]+)\)'
)
_COND_RE = re.compile(r"(p|c|len)\s*(<|<=|>|>=)\s*([\d.eE+-]+)")
_COEFF_RE = re.compile(r"(p|c|cr|cc)\s*\*\s*([\d.eE+-]+)")
_MONEY_RE = re.compile(r"[¥￥]\s*([\d.]+)")
_ABOVE_CNY_RE = re.compile(
    r"^(input|output|cache_read|cache_creation)_cny_per_1m_above_(\d+k?)_tokens$"
)
_LABEL_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kKmM]?)")


def cny_per_1m_to_usd_per_token(cny_per_1m: float, cny_per_usd: float) -> float:
    return round(cny_per_1m / cny_per_usd / 1_000_000, 12)


def round_cny(amount: float) -> float:
    # 与定价页展示对齐：≥1 两位小数，否则最多四位（¥0.2 / ¥2.5）
    if amount >= 1:
        return round(amount, 2)
    return round(amount, 4)


def is_price_field(key: str) -> bool:
    return key in _PRICE_FIELDS or bool(_ABOVE_CNY_RE.match(key))


def threshold_suffix(tokens: int) -> str:
    if tokens >= 1000 and tokens % 1000 == 0:
        return f"{tokens // 1000}k"
    return str(tokens)


def group_ratio(pricing: dict, group: str) -> float:
    """目标分组的倍率；default 分组总是返回 1.0（无需登录）。"""
    if group == "default":
        return 1.0
    value = (pricing.get("group_ratio") or {}).get(group)
    if value is None:
        raise ValueError(f"group_ratio[{group}] not found in pricing payload")
    return float(value)


def _parse_conditions(cond_str: str) -> list[dict[str, Any]]:
    if not cond_str:
        return []
    out = []
    for var, op, value in _COND_RE.findall(cond_str):
        out.append({"var": var, "op": op, "value": float(value)})
    return out


def _condition_upper(conditions: list[dict[str, Any]]) -> Optional[int]:
    """Upper exclusive-ish bound on prompt tokens for this tier (p < / <=)."""
    upper = None
    for cond in conditions:
        if cond["var"] not in ("p", "len"):
            continue
        if cond["op"] == "<=":
            upper = int(cond["value"]) if upper is None else min(upper, int(cond["value"]))
        elif cond["op"] == "<":
            # p < 32000 → 下一档按 above_32k（LiteLLM 为 prompt_tokens > 32000）
            v = int(cond["value"])
            upper = v if upper is None else min(upper, v)
    return upper


def _condition_lower(conditions: list[dict[str, Any]]) -> Optional[int]:
    """Lower bound where LiteLLM above_* applies (prompt_tokens > threshold)."""
    lower = None
    for cond in conditions:
        if cond["var"] not in ("p", "len"):
            continue
        if cond["op"] == ">":
            v = int(cond["value"])
            lower = v if lower is None else max(lower, v)
        elif cond["op"] == ">=":
            # >= N ≈ above_(N) 差 1 token；用 N-1 更贴，但阈值统一到千档
            v = max(0, int(cond["value"]) - 1)
            lower = v if lower is None else max(lower, v)
    return lower


def parse_all_tiers_usd(billing_expr: str) -> list[dict[str, Any]]:
    """Parse every tier(...) into {label, above_tokens|None, coeffs}."""
    raw_tiers: list[dict[str, Any]] = []
    for match in _TIER_RE.finditer(billing_expr or ""):
        cond_str, label, body = match.group(1) or "", match.group(2), match.group(3)
        coeffs = {k: float(v) for k, v in _COEFF_RE.findall(body)}
        if not coeffs:
            continue
        raw_tiers.append(
            {
                "label": label,
                "conditions": _parse_conditions(cond_str),
                "coeffs": coeffs,
            }
        )
    if not raw_tiers:
        return []

    parsed: list[dict[str, Any]] = []
    prev_upper: Optional[int] = None
    for idx, tier in enumerate(raw_tiers):
        if idx == 0:
            above = None
        else:
            above = _condition_lower(tier["conditions"])
            if above is None:
                above = prev_upper
        parsed.append(
            {
                "label": tier["label"],
                "above_tokens": above,
                "coeffs": tier["coeffs"],
            }
        )
        prev_upper = _condition_upper(tier["conditions"])
        if prev_upper is None and above is not None:
            prev_upper = above
    return parsed


def _coeffs_to_cny_fields(
    coeffs: dict[str, float],
    *,
    ratio: float,
    usd_exchange_rate: float,
    above_tokens: Optional[int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    suffix = (
        f"_above_{threshold_suffix(above_tokens)}_tokens"
        if above_tokens is not None
        else ""
    )
    for src, base_field in _TIER_FIELD_MAP.items():
        if src not in coeffs or coeffs[src] <= 0:
            continue
        out[f"{base_field}{suffix}"] = round_cny(
            coeffs[src] * ratio * usd_exchange_rate
        )
    return out


def model_to_cny_spec(
    model: dict,
    *,
    ratio: float,
    usd_exchange_rate: float,
) -> Optional[dict[str, Any]]:
    """Convert one /api/pricing row to rezecyan_prices.json model spec (CNY)."""
    endpoints = model.get("supported_endpoint_types") or []
    is_image = 1 == int(model.get("quota_type") or 0) or "image-generation" in endpoints

    if is_image:
        # UI: perCall = model_price * group_ratio; display CNY = perCall * usd_exchange_rate
        price = float(model.get("model_price") or 0) * ratio * usd_exchange_rate
        if price <= 0:
            return None
        return {
            "mode": "image_generation",
            "image_cny_per_image": round_cny(price),
            "pricing_group": TARGET_GROUP,
        }

    if model.get("billing_mode") == "tiered_expr" and model.get("billing_expr"):
        tiers = parse_all_tiers_usd(str(model["billing_expr"]))
        if not tiers:
            return None
        spec: dict[str, Any] = {
            "mode": "chat",
            "pricing_group": TARGET_GROUP,
        }
        cache_keys = False
        for tier in tiers:
            fields = _coeffs_to_cny_fields(
                tier["coeffs"],
                ratio=ratio,
                usd_exchange_rate=usd_exchange_rate,
                above_tokens=tier["above_tokens"],
            )
            spec.update(fields)
            if any(k.startswith("cache_") for k in fields):
                cache_keys = True
        if cache_keys:
            spec["supports_prompt_caching"] = True
        return spec if "input_cny_per_1m" in spec else None

    mr = float(model.get("model_ratio") or 0)
    if mr <= 0:
        return None
    # UI: inputPerM_USD = model_ratio * group_ratio * 2
    input_usd = mr * ratio * 2.0
    completion = float(model.get("completion_ratio") or 1.0)
    spec = {
        "mode": "chat",
        "pricing_group": TARGET_GROUP,
        "input_cny_per_1m": round_cny(input_usd * usd_exchange_rate),
        "output_cny_per_1m": round_cny(input_usd * completion * usd_exchange_rate),
    }
    if model.get("cache_ratio") not in (None, ""):
        spec["cache_read_cny_per_1m"] = round_cny(
            input_usd * float(model["cache_ratio"]) * usd_exchange_rate
        )
        spec["supports_prompt_caching"] = True
    if model.get("create_cache_ratio") not in (None, ""):
        spec["cache_creation_cny_per_1m"] = round_cny(
            input_usd * float(model["create_cache_ratio"]) * usd_exchange_rate
        )
        spec["supports_prompt_caching"] = True
    return spec


def dedupe_models(models: dict[str, dict]) -> dict[str, dict]:
    """Keep first occurrence per case-insensitive model id."""
    seen: set[str] = set()
    out: dict[str, dict] = {}
    for name, spec in models.items():
        key = name.strip()
        norm = key.lower()
        if not key or norm in seen:
            continue
        seen.add(norm)
        out[key] = spec
    return out


def build_entries(sheet: dict) -> dict:
    rate = sheet["cny_per_usd"]
    if not (isinstance(rate, (int, float)) and rate > 0):
        raise ValueError(f"invalid cny_per_usd: {rate!r}")

    entries: dict = {}
    for model, spec in dedupe_models(sheet["models"]).items():
        cny_audit = {k: v for k, v in spec.items() if is_price_field(k) and k != "image_cny_per_image"}
        entry: dict = {
            "litellm_provider": "rezecyan",
            "provider_pricing_currency": "USD",
            "source": spec.get("source") or sheet.get("source", ""),
            "original_pricing_cny_per_1m": cny_audit,
            "cny_per_usd_rate": rate,
            "price_as_of": sheet.get("as_of", ""),
        }
        if not entry["original_pricing_cny_per_1m"]:
            del entry["original_pricing_cny_per_1m"]
        for cny_field, usd_field in CNY_PER_1M_FIELD_MAP.items():
            if cny_field in spec:
                entry[usd_field] = cny_per_1m_to_usd_per_token(spec[cny_field], rate)
        for cny_field, usd_field in CNY_PER_ITEM_FIELD_MAP.items():
            if cny_field in spec:
                entry[usd_field] = round(spec[cny_field] / rate, 12)
        for cny_field, value in spec.items():
            match = _ABOVE_CNY_RE.match(cny_field)
            if not match:
                continue
            kind, thresh = match.group(1), match.group(2)
            base = _ABOVE_USD_FIELD[f"{kind}_cny_per_1m"]
            entry[f"{base}_above_{thresh}_tokens"] = cny_per_1m_to_usd_per_token(
                value, rate
            )
        for k, v in spec.items():
            if is_price_field(k) or k in {"source", "pricing_group"}:
                continue
            entry[k] = v
        entries[f"rezecyan/{model}"] = entry
    return entries


# 匹配一个顶层 "rezecyan/..." 条目块（4 空格缩进；嵌套对象闭合在更深缩进，不会提前命中）
_ENTRY_BLOCK_RE = re.compile(r'    "rezecyan/[^"]*": \{.*?\n    \},?\n', re.DOTALL)


def _render_entry_block(entries: dict) -> str:
    lines = []
    for key in sorted(entries):
        body = json.dumps(entries[key], indent=4, ensure_ascii=False, sort_keys=True)
        indented = "\n".join(
            "    " + line if line else line for line in body.splitlines()
        )
        lines.append(f'    "{key}": ' + indented.lstrip())
    return ",\n".join(lines)


def splice_entries(raw: str, entries: dict) -> str:
    """删除全部旧 rezecyan/* 块再追加（去重），文本级操作避免整文件重写产生无关 diff。"""
    raw = _ENTRY_BLOCK_RE.sub("", raw)
    raw = re.sub(r",(\s*)\n\}", r"\1\n}", raw.rstrip() + "\n")
    end = raw.rindex("\n}")
    block = _render_entry_block(entries)
    return raw[:end] + ",\n" + block + raw[end:]


def sync(dry_run: bool, sheet: Optional[dict] = None) -> dict:
    data: dict = json.loads(SOURCE_SHEET.read_text()) if sheet is None else sheet
    entries = build_entries(data)
    if dry_run:
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return entries

    for target in TARGET_FILES:
        raw = target.read_text()
        updated = splice_entries(raw, entries)
        parsed = json.loads(updated)
        # 去重断言：生成键唯一，且文件中不再残留其它 rezecyan/*
        rez_keys = [k for k in parsed if k.startswith("rezecyan/")]
        assert len(rez_keys) == len(set(rez_keys)) == len(entries), rez_keys
        for key in entries:
            assert parsed[key]["litellm_provider"] == "rezecyan", key
        target.write_text(updated)
        print(f"wrote {len(entries)} rezecyan entries -> {target}")
    return entries


def _parse_money(text: str) -> Optional[float]:
    match = _MONEY_RE.search(text or "")
    return float(match.group(1)) if match else None


def _label_token_bounds(label: str) -> tuple[Optional[int], Optional[int]]:
    """Parse '[0~256k]' / '[256k~1m]' style badge into (lower, upper) tokens."""
    nums = []
    for raw, unit in _LABEL_NUM_RE.findall(label or ""):
        value = float(raw)
        u = unit.lower()
        if u == "k":
            value *= 1000
        elif u == "m":
            value *= 1_000_000
        nums.append(int(value))
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], None
    return None, None


def _parse_price_cols(root) -> dict[str, float]:
    label_map = {
        "输入": "input_cny_per_1m",
        "输出": "output_cny_per_1m",
        "缓存读取": "cache_read_cny_per_1m",
        "缓存写入": "cache_creation_cny_per_1m",
        "按次计费": "image_cny_per_image",
    }
    out: dict[str, float] = {}
    for col in root.query_selector_all(".rz-detail-price-col"):
        if "rz-detail-price-col--muted" in (col.get_attribute("class") or ""):
            continue
        label_el = col.query_selector(".rz-detail-price-col__k")
        value_el = col.query_selector(".rz-detail-price-col__v")
        if not label_el or not value_el:
            continue
        field = label_map.get(label_el.inner_text().strip())
        if not field:
            continue
        amount = _parse_money(value_el.inner_text())
        if amount is not None:
            out[field] = amount
    return out


def _scrape_group_li_prices(group_li) -> dict[str, float]:
    """Parse CNY prices from one .rz-detail-group; include every tier panel."""
    tiers = group_li.query_selector_all(".rz-detail-tier")
    if not tiers:
        return _parse_price_cols(group_li)

    out: dict[str, float] = {}
    prev_upper: Optional[int] = None
    for idx, tier in enumerate(tiers):
        prices = _parse_price_cols(tier)
        if not prices:
            continue
        badge = tier.query_selector(".rz-detail-tier__badge")
        lower, upper = _label_token_bounds(badge.inner_text() if badge else "")
        if idx == 0:
            out.update(prices)
        else:
            above = lower if lower and lower > 0 else prev_upper
            if above is None:
                continue
            suffix = threshold_suffix(above)
            for field, amount in prices.items():
                out[f"{field}_above_{suffix}_tokens"] = amount
        if upper is not None:
            prev_upper = upper
        elif lower is not None and idx > 0:
            prev_upper = lower
    return out


def _click_model_prices(page, model_name: str) -> dict[str, float]:
    """Click model card and read prices from the default group panel.
    
    default 分组总是第一个面板，无需特殊查找逻辑。
    """
    card = page.query_selector(f'article.rz-model-card[data-model="{model_name}"]')
    if not card:
        return {}
    card.click(timeout=5000)
    page.wait_for_selector(".rz-modal:not([hidden]) .rz-detail-groups, .rz-modal__inner .rz-detail-group", timeout=15000)
    # default 分组通常是第一个 .rz-detail-group
    first_group = page.query_selector(".rz-modal .rz-detail-group")
    prices = _scrape_group_li_prices(first_group) if first_group else {}
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    return prices


def scrape_rezecyan_prices(headless: bool = True, timeout_ms: int = 120000) -> dict:
    """Playwright: open pricing page, scrape default group prices (no login required)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from e

    existing = json.loads(SOURCE_SHEET.read_text()) if SOURCE_SHEET.exists() else {}
    print(f"Scraping {PRICING_URL} (group={TARGET_GROUP}, no login required)...", file=sys.stderr)

    pricing_payload: Optional[dict] = None
    status_payload: Optional[dict] = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        # default 分组无需登录，直接使用新 context
        context = browser.new_context()
        page = context.new_page()

        def on_response(response) -> None:
            nonlocal pricing_payload, status_payload
            url = response.url
            if response.status != 200:
                return
            try:
                if url.rstrip("/").endswith("/api/pricing"):
                    pricing_payload = response.json()
                elif url.rstrip("/").endswith("/api/status"):
                    body = response.json()
                    status_payload = body.get("data", body)
            except Exception:
                return

        page.on("response", on_response)
        page.goto(PRICING_URL, wait_until="networkidle", timeout=timeout_ms)
        page.wait_for_selector("article.rz-model-card[data-model]", timeout=timeout_ms)
        # 等待 pricing API 响应
        for _ in range(50):
            if pricing_payload and pricing_payload.get("data"):
                break
            page.wait_for_timeout(100)
        if not pricing_payload or not pricing_payload.get("data"):
            browser.close()
            raise RuntimeError("未能拦截 /api/pricing；页面结构或接口可能已变")

        usd_rate = float((status_payload or {}).get("usd_exchange_rate") or 0) or float(
            existing.get("cny_per_usd") or 7.2
        )
        ratio = group_ratio(pricing_payload, TARGET_GROUP)
        print(
            f"usd_exchange_rate={usd_rate} group_ratio[{TARGET_GROUP}]={ratio}",
            file=sys.stderr,
        )

        candidates = []
        seen_names: set[str] = set()
        for row in pricing_payload["data"]:
            name = str(row.get("model_name") or "").strip()
            if not name:
                continue
            groups = row.get("enable_groups") or []
            if TARGET_GROUP not in groups:
                continue
            # API 侧去重：同名只保留第一条
            if name.lower() in seen_names:
                print(f"  skip duplicate api row: {name}", file=sys.stderr)
                continue
            seen_names.add(name.lower())
            candidates.append(row)

        print(f"{len(candidates)} models enable {TARGET_GROUP}", file=sys.stderr)
        scraped: dict[str, dict] = {}
        for row in candidates:
            name = row["model_name"]
            computed = model_to_cny_spec(row, ratio=ratio, usd_exchange_rate=usd_rate)
            if not computed:
                print(f"  skip unpriced: {name}", file=sys.stderr)
                continue
            try:
                clicked = _click_model_prices(page, name)
                if not clicked:
                    print(
                        f"  {name}: modal 无价格面板，采用 api 公式价",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"  click failed {name}: {exc}; using api formula", file=sys.stderr)
                clicked = {}

            spec = dict(computed)
            # 弹窗价格覆盖同名字段（含 above_* 分档）
            for field, amount in clicked.items():
                if is_price_field(field):
                    spec[field] = amount
            if "image_cny_per_image" in spec:
                spec["mode"] = "image_generation"
            else:
                spec["mode"] = spec.get("mode") or "chat"

            # 保留手工维护的非价格字段（max_tokens / supports_* 等）
            prev = (existing.get("models") or {}).get(name) or {}
            for k, v in prev.items():
                if not is_price_field(k) and k not in spec:
                    spec[k] = v

            scraped[name] = spec
            above_n = sum(1 for k in spec if "_above_" in k)
            print(
                f"  + {name} input={spec.get('input_cny_per_1m')} "
                f"output={spec.get('output_cny_per_1m')} "
                f"image={spec.get('image_cny_per_image')} "
                f"above_fields={above_n}",
                file=sys.stderr,
            )

        browser.close()

    scraped = dedupe_models(scraped)
    if not scraped:
        raise RuntimeError(f"No models scraped for group {TARGET_GROUP}")

    return {
        "_comment": existing.get(
            "_comment",
            "Rezecyan 人民币牌价源文件（唯一维护入口）。改价后运行 sync_rezecyan_prices.py。",
        ),
        "cny_per_usd": usd_rate,
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "source": PRICING_URL,
        "pricing_group": TARGET_GROUP,
        "models": scraped,
    }


def save_login_state(timeout_ms: int = 600000) -> None:
    """打开可见浏览器让用户登录，保存 storage state 供 --scrape 复用。"""
    from playwright.sync_api import sync_playwright

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=timeout_ms)
        print("请在弹出的浏览器中完成登录，然后回到终端按回车保存登录态…", file=sys.stderr)
        input()
        context.storage_state(path=str(STATE_FILE))
        browser.close()
    print(f"登录态已保存到 {STATE_FILE}", file=sys.stderr)


def self_test() -> None:
    assert cny_per_1m_to_usd_per_token(2.0, 7.2) == round(2.0 / 7.2 / 1e6, 12)
    # default 分组总是返回 1.0
    assert group_ratio({"group_ratio": {"default": 1}}, "default") == 1.0
    assert group_ratio({"group_ratio": {}}, "default") == 1.0
    assert group_ratio({}, "default") == 1.0

    tiers = parse_all_tiers_usd(
        'p <= 256000 ? tier("[0~256k]", p * 0.285714 + c * 1.142857 + cr * 0.028571 + cc * 0.357142) '
        ": tier(\"[256k~1m]\", p * 0.857142 + c * 3.428571 + cr * 0.085714 + cc * 1.071428)"
    )
    assert len(tiers) == 2
    assert tiers[0]["above_tokens"] is None
    assert tiers[1]["above_tokens"] == 256000
    assert abs(tiers[0]["coeffs"]["p"] - 0.285714) < 1e-9
    assert abs(tiers[1]["coeffs"]["p"] - 0.857142) < 1e-9

    triple = parse_all_tiers_usd(
        "p <= 32000 ? tier(\"[0~32k]\", p * 0.142857 + c * 1.428571) : "
        "p > 32000 && p <= 128000 ? tier(\"[32k~128k]\", p * 0.214285 + c * 2.142857) : "
        'tier("[128k~256k]", p * 0.428571 + c * 4.285714)'
    )
    assert [t["above_tokens"] for t in triple] == [None, 32000, 128000]

    qwen = model_to_cny_spec(
        {
            "model_name": "qwen3.7-plus",
            "quota_type": 0,
            "billing_mode": "tiered_expr",
            "billing_expr": (
                'p <= 256000 ? tier("[0~256k]", p * 0.285714 + c * 1.142857 + '
                "cr * 0.028571 + cc * 0.357142) : "
                'tier("[256k~1m]", p * 0.857142 + c * 3.428571 + cr * 0.085714 + cc * 1.071428)'
            ),
            "enable_groups": ["ali-of-pro"],
        },
        ratio=1.0,
        usd_exchange_rate=7.0,
    )
    assert qwen is not None
    assert qwen["input_cny_per_1m"] == 2.0
    assert qwen["output_cny_per_1m"] == 8.0
    assert qwen["cache_read_cny_per_1m"] == 0.2
    assert qwen["cache_creation_cny_per_1m"] == 2.5
    assert qwen["input_cny_per_1m_above_256k_tokens"] == 6.0
    assert qwen["output_cny_per_1m_above_256k_tokens"] == 24.0
    assert qwen["cache_read_cny_per_1m_above_256k_tokens"] == 0.6
    assert qwen["cache_creation_cny_per_1m_above_256k_tokens"] == 7.5

    flat = model_to_cny_spec(
        {
            "model_name": "x",
            "quota_type": 0,
            "model_ratio": 0.15,
            "completion_ratio": 4,
            "cache_ratio": 0.2,
        },
        ratio=1.0,
        usd_exchange_rate=7.0,
    )
    assert flat is not None
    assert flat["input_cny_per_1m"] == 2.1  # 0.15*2*7
    assert flat["output_cny_per_1m"] == 8.4

    assert dedupe_models({"A": {"mode": "chat"}, "a": {"mode": "image"}, "B": {"mode": "chat"}}) == {
        "A": {"mode": "chat"},
        "B": {"mode": "chat"},
    }

    entries = build_entries(
        {
            "cny_per_usd": 7.0,
            "as_of": "2026-07-28",
            "source": "https://example.com",
            "models": {
                "m1": {
                    "mode": "chat",
                    "input_cny_per_1m": 2.0,
                    "output_cny_per_1m": 8.0,
                    "cache_read_cny_per_1m": 0.2,
                    "cache_creation_cny_per_1m": 2.5,
                    "input_cny_per_1m_above_256k_tokens": 6.0,
                    "output_cny_per_1m_above_256k_tokens": 24.0,
                    "supports_reasoning": True,
                }
            },
        }
    )
    e = entries["rezecyan/m1"]
    assert e["litellm_provider"] == "rezecyan"
    assert e["input_cost_per_token"] == round(2.0 / 7.0 / 1e6, 12)
    assert e["input_cost_per_token_above_256k_tokens"] == round(6.0 / 7.0 / 1e6, 12)
    assert e["output_cost_per_token_above_256k_tokens"] == round(24.0 / 7.0 / 1e6, 12)
    assert e["supports_reasoning"] is True
    assert "pricing_group" not in e
    assert "input_cny_per_1m_above_256k_tokens" not in e

    raw = '{\n    "other/model": {\n        "mode": "chat"\n    }\n}\n'
    once = splice_entries(raw, entries)
    parsed = json.loads(once)
    assert "rezecyan/m1" in parsed and "other/model" in parsed
    twice = splice_entries(once, entries)
    assert json.loads(twice) == parsed
    # 人为塞入重复块后 splice 仍只剩一份
    duped = once.replace(
        '    "rezecyan/m1"',
        '    "rezecyan/m1": {\n        "mode": "chat"\n    },\n    "rezecyan/m1"',
        1,
    )
    cleaned = json.loads(splice_entries(duped, entries))
    assert len([k for k in cleaned if k.startswith("rezecyan/")]) == 1
    print("self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing JSON")
    parser.add_argument("--self-test", action="store_true", help="Run internal tests")
    parser.add_argument(
        "--scrape",
        action="store_true",
        help=f"Playwright scrape {PRICING_URL} (only {TARGET_GROUP}), then sync price JSONs",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window during scraping",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help=f"打开浏览器手动登录并保存登录态到 {STATE_FILE}（default 分组无需此步骤）",
    )
    parser.add_argument(
        "--sheet-only",
        action="store_true",
        help="With --scrape: only write rezecyan_prices.json, do not sync model_prices JSONs",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        sys.exit(0)

    if args.login:
        save_login_state()
        sys.exit(0)

    if args.scrape:
        scraped = scrape_rezecyan_prices(headless=not args.no_headless)
        if args.dry_run:
            print(json.dumps(scraped, indent=2, ensure_ascii=False))
        else:
            SOURCE_SHEET.write_text(
                json.dumps(scraped, indent=2, ensure_ascii=False) + "\n"
            )
            print(f"wrote {SOURCE_SHEET} ({len(scraped['models'])} models)", file=sys.stderr)
            if not args.sheet_only:
                sync(dry_run=False, sheet=scraped)
    else:
        sync(dry_run=args.dry_run)
