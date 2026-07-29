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
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHEET = REPO_ROOT / "scripts" / "rezecyan_prices.json"
TARGET_FILES = [
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
]

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


def cny_per_1m_to_usd_per_token(cny_per_1m: float, cny_per_usd: float) -> float:
    return round(cny_per_1m / cny_per_usd / 1_000_000, 12)


def build_entries(sheet: dict) -> dict:
    rate = sheet["cny_per_usd"]
    if not (isinstance(rate, (int, float)) and rate > 0):
        raise ValueError(f"invalid cny_per_usd: {rate!r}")

    entries: dict = {}
    for model, spec in sheet["models"].items():
        entry: dict = {
            "litellm_provider": "rezecyan",
            "provider_pricing_currency": "USD",
            "source": spec.get("source") or sheet.get("source", ""),
            # 审计字段：人民币原价 + 所用汇率 + 基准日期，供对账与回溯
            "original_pricing_cny_per_1m": {
                k: v for k, v in spec.items() if k in CNY_PER_1M_FIELD_MAP
            },
            "cny_per_usd_rate": rate,
            "price_as_of": sheet.get("as_of", ""),
        }
        for cny_field, usd_field in CNY_PER_1M_FIELD_MAP.items():
            if cny_field in spec:
                entry[usd_field] = cny_per_1m_to_usd_per_token(spec[cny_field], rate)
        for cny_field, usd_field in CNY_PER_ITEM_FIELD_MAP.items():
            if cny_field in spec:
                entry[usd_field] = round(spec[cny_field] / rate, 12)
        # 其余字段（mode/max_tokens/supports_* 等）原样透传
        passthrough_excludes = (
            set(CNY_PER_1M_FIELD_MAP) | set(CNY_PER_ITEM_FIELD_MAP) | {"source"}
        )
        for k, v in spec.items():
            if k not in passthrough_excludes:
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
    """删除旧 rezecyan/* 块并在顶层对象末尾追加新块（文本级操作，避免整文件重写产生无关 diff）。"""
    raw = _ENTRY_BLOCK_RE.sub("", raw)
    # 清理删除后可能残留的悬挂逗号（被删块位于文件末尾的情形）
    raw = re.sub(r",(\s*)\n\}", r"\1\n}", raw.rstrip() + "\n")
    end = raw.rindex("\n}")
    block = _render_entry_block(entries)
    return raw[:end] + ",\n" + block + raw[end:]


def sync(dry_run: bool) -> None:
    sheet = json.loads(SOURCE_SHEET.read_text())
    entries = build_entries(sheet)
    if dry_run:
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return

    for target in TARGET_FILES:
        raw = target.read_text()
        updated = splice_entries(raw, entries)
        parsed = json.loads(updated)  # 写盘前校验 JSON 合法性
        for key in entries:
            assert parsed[key]["litellm_provider"] == "rezecyan", key
        target.write_text(updated)
        print(f"wrote {len(entries)} rezecyan entries -> {target}")


def self_test() -> None:
    assert cny_per_1m_to_usd_per_token(2.0, 7.2) == round(2.0 / 7.2 / 1e6, 12)
    entries = build_entries(
        {
            "cny_per_usd": 7.2,
            "as_of": "2026-07-28",
            "source": "https://example.com",
            "models": {
                "m1": {
                    "mode": "chat",
                    "input_cny_per_1m": 2.0,
                    "output_cny_per_1m": 8.0,
                    "cache_read_cny_per_1m": 0.2,
                    "cache_creation_cny_per_1m": 2.5,
                    "supports_reasoning": True,
                }
            },
        }
    )
    e = entries["rezecyan/m1"]
    assert e["litellm_provider"] == "rezecyan"
    assert e["input_cost_per_token"] == round(2.0 / 7.2 / 1e6, 12)
    assert e["output_cost_per_token"] == round(8.0 / 7.2 / 1e6, 12)
    assert e["cache_read_input_token_cost"] == round(0.2 / 7.2 / 1e6, 12)
    assert e["cache_creation_input_token_cost"] == round(2.5 / 7.2 / 1e6, 12)
    assert e["supports_reasoning"] is True
    assert e["cny_per_usd_rate"] == 7.2
    assert "input_cny_per_1m" not in e  # CNY 字段不得泄漏到生成条目顶层
    assert e["original_pricing_cny_per_1m"]["input_cny_per_1m"] == 2.0

    # splice：新增、幂等重放、结尾块清理均不破坏 JSON
    raw = '{\n    "other/model": {\n        "mode": "chat"\n    }\n}\n'
    once = splice_entries(raw, entries)
    parsed = json.loads(once)
    assert "rezecyan/m1" in parsed and "other/model" in parsed
    twice = splice_entries(once, entries)
    assert json.loads(twice) == parsed
    print("self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        sys.exit(0)
    sync(dry_run=args.dry_run)
