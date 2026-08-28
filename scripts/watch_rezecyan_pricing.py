#!/usr/bin/env python3
"""
Watch Rezecyan GET /api/pricing (no cookie). Compare against a local snapshot.

只监视、不改价：不写 rezecyan_prices.json，不写 model_prices JSON，不跑 Playwright。

    python3 scripts/watch_rezecyan_pricing.py --init       # 用当前接口生成/覆盖快照
    python3 scripts/watch_rezecyan_pricing.py              # 对比；有漂移发飞书
    python3 scripts/watch_rezecyan_pricing.py --no-feishu  # 只打印
    python3 scripts/watch_rezecyan_pricing.py --self-test

飞书：环境变量 FEISHU_WEBHOOK_URL（bot v2 webhook）。未配置则只打印。

价格对比：billing_expr 原文 + 各档 CNY（公式系数 × 源表汇率），不执行 hour()。
公式仅大小写变化（如 tier("base") → tier("BASE")）且各档 CNY 不变：不当漂移、不发飞书，并回写快照。
分组：盯 watched_groups 的进出；模型新出现任意组名也报；非盯梢组被删不报。
游客列表消失 = default 下架（公开列表不再包含）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sync_rezecyan_prices import (
    SOURCE_SHEET,
    is_price_field,
    model_to_cny_spec,
)

SCRIPTS_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = SCRIPTS_DIR / "rezecyan_pricing_snapshot.json"
PRICING_API = "https://cn.rezecyan.com/api/pricing"

# 开发机当前在用的令牌分组（截图）。可在快照 JSON 里改，不必改代码。
DEFAULT_WATCHED_GROUPS = [
    "default",
    "tx-glm53",
    "ali-dspro813",
    "ali-dsflash731",
    "idc-k3-svip",
    "ali-glm-svip",
    "ali-vip-svip",
]

_RATIO_KEYS = (
    "quota_type",
    "model_ratio",
    "completion_ratio",
    "cache_ratio",
    "create_cache_ratio",
    "model_price",
)


def load_cny_per_usd() -> float:
    sheet = json.loads(SOURCE_SHEET.read_text())
    rate = float(sheet.get("cny_per_usd") or 0)
    if rate <= 0:
        raise ValueError(f"invalid cny_per_usd in {SOURCE_SHEET}")
    return rate


def fetch_pricing(timeout: int = 30) -> dict:
    req = urllib.request.Request(
        PRICING_API,
        headers={"accept": "application/json", "user-agent": "rezecyan-price-watch/1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload.get("success", True):
        raise RuntimeError(f"pricing API success=false: {payload}")
    if not isinstance(payload.get("data"), list):
        raise RuntimeError("pricing API missing data[]")
    return payload


def _num(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def row_to_record(row: dict, *, usd_rate: float) -> dict[str, Any]:
    groups = sorted({str(g) for g in (row.get("enable_groups") or []) if g})
    expr = str(row.get("billing_expr") or "").strip()
    mode = str(row.get("billing_mode") or "").strip() or ("tiered_expr" if expr else "ratio")
    spec = model_to_cny_spec(row, ratio=1.0, usd_exchange_rate=usd_rate) or {}
    rec: dict[str, Any] = {
        "enable_groups": groups,
        "quota_type": int(row.get("quota_type") or 0),
        "billing_mode": mode,
        "billing_expr": expr,
        "model_ratio": _num(row.get("model_ratio")),
        "completion_ratio": _num(row.get("completion_ratio")),
        "model_price": _num(row.get("model_price")),
    }
    if row.get("cache_ratio") not in (None, ""):
        rec["cache_ratio"] = _num(row["cache_ratio"])
    if row.get("create_cache_ratio") not in (None, ""):
        rec["create_cache_ratio"] = _num(row["create_cache_ratio"])
    for key, value in spec.items():
        if is_price_field(key):
            rec[key] = value
    return rec


def payload_to_snapshot(payload: dict, *, usd_rate: float, watched: list[str]) -> dict:
    models: dict[str, dict] = {}
    seen: set[str] = set()
    for row in payload.get("data") or []:
        name = str(row.get("model_name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        models[name] = row_to_record(row, usd_rate=usd_rate)
    return {
        "_comment": (
            "Rezecyan /api/pricing 监视快照（只给 watch_rezecyan_pricing.py 用）。"
            "确认漂移后用 --init 接受为新基线。不要当 LiteLLM 计费表。"
        ),
        "pricing_api": PRICING_API,
        "watched_groups": list(watched),
        "cny_per_usd": usd_rate,
        "as_of": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "models": models,
    }


def _price_keys(record: dict) -> list[str]:
    return sorted(k for k in record if is_price_field(k))


def _expr_case_only(old: str, new: str) -> bool:
    """True when formulas differ only by letter case (tier labels, hour, …)."""
    a, b = (old or "").strip(), (new or "").strip()
    return bool(a) and bool(b) and a != b and a.casefold() == b.casefold()


def _price_diff(old: dict, new: dict) -> dict[str, Any]:
    changed: dict[str, Any] = {}
    old_expr = old.get("billing_expr") or ""
    new_expr = new.get("billing_expr") or ""
    if old_expr != new_expr and not _expr_case_only(old_expr, new_expr):
        changed["billing_expr"] = {"old": old_expr, "new": new_expr}
    if (old.get("billing_mode") or "") != (new.get("billing_mode") or ""):
        changed["billing_mode"] = {"old": old.get("billing_mode"), "new": new.get("billing_mode")}
    keys = set(_price_keys(old)) | set(_price_keys(new))
    for key in sorted(keys):
        if old.get(key) != new.get(key):
            changed[key] = {"old": old.get(key), "new": new.get(key)}
    # 无公式时倍率/按次价才当价格信号；有 billing_expr 时忽略残留 model_ratio
    if not (new.get("billing_expr") or old.get("billing_expr")):
        for key in _RATIO_KEYS:
            if _num(old.get(key)) != _num(new.get(key)):
                changed[key] = {"old": old.get(key), "new": new.get(key)}
    return changed


def diff_snapshots(old: dict, new: dict) -> dict[str, Any]:
    watched = set(old.get("watched_groups") or new.get("watched_groups") or DEFAULT_WATCHED_GROUPS)
    old_models = old.get("models") or {}
    new_models = new.get("models") or {}
    old_names = set(old_models)
    new_names = set(new_models)

    added = []
    for name in sorted(new_names - old_names):
        rec = new_models[name]
        added.append(
            {
                "model": name,
                "enable_groups": rec.get("enable_groups") or [],
                "watched_groups": sorted(set(rec.get("enable_groups") or []) & watched),
            }
        )
    removed = []
    for name in sorted(old_names - new_names):
        rec = old_models[name]
        removed.append(
            {
                "model": name,
                "note": "游客 /api/pricing 不再返回（default 下架 / 公开列表不再包含）",
                "enable_groups_was": rec.get("enable_groups") or [],
            }
        )

    price_changes = []
    group_changes = []
    ignored_expr_case = []
    for name in sorted(old_names & new_names):
        prev, curr = old_models[name], new_models[name]
        priced = _price_diff(prev, curr)
        if priced:
            price_changes.append({"model": name, "fields": priced})
        elif _expr_case_only(prev.get("billing_expr") or "", curr.get("billing_expr") or ""):
            ignored_expr_case.append(name)

        old_g = set(prev.get("enable_groups") or [])
        new_g = set(curr.get("enable_groups") or [])
        added_g = new_g - old_g
        removed_g = old_g - new_g
        watched_added = sorted(added_g & watched)
        watched_removed = sorted(removed_g & watched)
        other_added = sorted(added_g - watched)
        if watched_added or watched_removed or other_added:
            group_changes.append(
                {
                    "model": name,
                    "watched_added": watched_added,
                    "watched_removed": watched_removed,
                    "other_added": other_added,
                }
            )

    return {
        "added": added,
        "removed": removed,
        "price_changes": price_changes,
        "group_changes": group_changes,
        "ignored_expr_case": ignored_expr_case,
    }


def has_drift(diff: dict) -> bool:
    return bool(
        diff["added"] or diff["removed"] or diff["price_changes"] or diff["group_changes"]
    )


# 该 webhook 在飞书侧开了「自定义关键词」，正文必须包含此词，否则 code=19024 且群里无消息。
FEISHU_KEYWORD = "告警"


def format_report(diff: dict) -> str:
    lines = [f"{FEISHU_KEYWORD} Reze 定价监视：发现漂移（只通知，未改价）", ""]
    if diff["added"]:
        lines.append("【上新】游客接口新出现：")
        for item in diff["added"]:
            groups = ", ".join(item["enable_groups"]) or "-"
            watched = ", ".join(item["watched_groups"]) or "无盯梢组"
            lines.append(f"  + {item['model']}  分组[{groups}]  盯梢命中[{watched}]")
        lines.append("")
    if diff["removed"]:
        lines.append("【default 下架】公开列表不再包含：")
        for item in diff["removed"]:
            was = ", ".join(item["enable_groups_was"]) or "-"
            lines.append(f"  - {item['model']}  原先分组[{was}]")
        lines.append("")
    if diff["price_changes"]:
        lines.append("【改价】公式或各档牌价变化（非夜间此刻价）：")
        for item in diff["price_changes"]:
            lines.append(f"  * {item['model']}")
            for field, pair in item["fields"].items():
                old, new = pair["old"], pair["new"]
                if field == "billing_expr":
                    lines.append(f"      billing_expr 已变")
                    lines.append(f"        was: {old}")
                    lines.append(f"        now: {new}")
                else:
                    lines.append(f"      {field}: {old} → {new}")
        lines.append("")
    if diff["group_changes"]:
        lines.append("【分组】盯梢组进出，或出现新组名：")
        for item in diff["group_changes"]:
            parts = []
            if item["watched_added"]:
                parts.append("盯梢组加入 " + ",".join(item["watched_added"]))
            if item["watched_removed"]:
                parts.append("盯梢组去掉 " + ",".join(item["watched_removed"]))
            if item["other_added"]:
                parts.append("新组名 " + ",".join(item["other_added"]))
            lines.append(f"  * {item['model']}: {'; '.join(parts)}")
        lines.append("")
    lines.append("确认后执行: python3 scripts/watch_rezecyan_pricing.py --init")
    lines.append("Apply 仍是人审后跑 sync_rezecyan_prices.py --scrape / 打镜像 / ERP")
    return "\n".join(lines).rstrip() + "\n"


def send_feishu(text: str, webhook: str) -> None:
    if FEISHU_KEYWORD not in text:
        text = f"{FEISHU_KEYWORD} {text}"
    body = json.dumps({"msg_type": "text", "content": {"text": text}}, ensure_ascii=False).encode()
    req = urllib.request.Request(
        webhook,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode()
    if not raw.strip():
        return
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"feishu 响应不是 JSON: {raw[:200]}") from exc
    code = parsed.get("code", 0)
    if code not in (0, None):
        raise RuntimeError(f"feishu code={code} msg={parsed.get('msg')}")


def write_snapshot(snapshot: dict, path: Path = SNAPSHOT_PATH) -> None:
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")


def self_test() -> None:
    old = {
        "watched_groups": ["default", "ali-vip-svip"],
        "models": {
            "keep": {
                "enable_groups": ["default", "ali-vip-svip", "noise"],
                "billing_mode": "tiered_expr",
                "billing_expr": 'tier("base", p * 0.1 + c * 0.2)',
                "input_cny_per_1m": 0.7,
                "output_cny_per_1m": 1.4,
                "model_ratio": 37.5,
            },
            "gone": {"enable_groups": ["default"], "billing_expr": "", "input_cny_per_1m": 1},
        },
    }
    new = {
        "watched_groups": ["default", "ali-vip-svip"],
        "models": {
            "keep": {
                "enable_groups": ["default", "brand-new"],
                "billing_mode": "tiered_expr",
                "billing_expr": 'tier("base", p * 0.2 + c * 0.2)',
                "input_cny_per_1m": 1.4,
                "output_cny_per_1m": 1.4,
                "model_ratio": 99,
            },
            "fresh": {"enable_groups": ["default", "ali-vip-svip"], "billing_expr": "", "input_cny_per_1m": 2},
        },
    }
    diff = diff_snapshots(old, new)
    assert [x["model"] for x in diff["added"]] == ["fresh"]
    assert [x["model"] for x in diff["removed"]] == ["gone"]
    assert diff["price_changes"][0]["model"] == "keep"
    assert "billing_expr" in diff["price_changes"][0]["fields"]
    assert "input_cny_per_1m" in diff["price_changes"][0]["fields"]
    assert "model_ratio" not in diff["price_changes"][0]["fields"]
    g = diff["group_changes"][0]
    assert g["model"] == "keep"
    assert g["watched_removed"] == ["ali-vip-svip"]
    assert g["other_added"] == ["brand-new"]
    assert "noise" not in json.dumps(g)

    case_old = {
        "watched_groups": ["default"],
        "models": {
            "flash": {
                "enable_groups": ["default"],
                "billing_mode": "tiered_expr",
                "billing_expr": 'tier("base", p * 0.42857142857 + c * 1.28571428571)',
                "input_cny_per_1m": 3.0,
                "output_cny_per_1m": 9.0,
            }
        },
    }
    case_new = {
        "watched_groups": ["default"],
        "models": {
            "flash": {
                "enable_groups": ["default"],
                "billing_mode": "tiered_expr",
                "billing_expr": 'tier("BASE", p * 0.42857142857 + c * 1.28571428571)',
                "input_cny_per_1m": 3.0,
                "output_cny_per_1m": 9.0,
            }
        },
    }
    case_diff = diff_snapshots(case_old, case_new)
    assert case_diff["price_changes"] == []
    assert case_diff["ignored_expr_case"] == ["flash"]
    assert not has_drift(case_diff)

    mixed = json.loads(json.dumps(case_new))
    mixed["models"]["flash"]["input_cny_per_1m"] = 4.0
    mixed_diff = diff_snapshots(case_old, mixed)
    assert "billing_expr" not in mixed_diff["price_changes"][0]["fields"]
    assert mixed_diff["price_changes"][0]["fields"]["input_cny_per_1m"]["new"] == 4.0
    assert mixed_diff["ignored_expr_case"] == []
    print("self-test OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", action="store_true", help="用当前接口覆盖快照（接受基线）")
    parser.add_argument("--no-feishu", action="store_true", help="有漂移也不发飞书")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    usd_rate = load_cny_per_usd()
    try:
        payload = fetch_pricing()
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"拉取 {PRICING_API} 失败: {exc}", file=sys.stderr)
        return 2

    watched = DEFAULT_WATCHED_GROUPS
    if args.snapshot.exists() and not args.init:
        existing = json.loads(args.snapshot.read_text())
        watched = existing.get("watched_groups") or DEFAULT_WATCHED_GROUPS

    live = payload_to_snapshot(payload, usd_rate=usd_rate, watched=watched)

    if args.init or not args.snapshot.exists():
        write_snapshot(live, args.snapshot)
        print(f"wrote {args.snapshot} ({len(live['models'])} models, rate={usd_rate})")
        return 0

    baseline = json.loads(args.snapshot.read_text())
    live["watched_groups"] = baseline.get("watched_groups") or watched
    diff = diff_snapshots(baseline, live)
    if not has_drift(diff):
        ignored = diff.get("ignored_expr_case") or []
        if ignored:
            write_snapshot(live, args.snapshot)
            print(
                f"no drift ({len(live['models'])} models); "
                f"absorbed case-only billing_expr: {', '.join(ignored)}"
            )
        else:
            print(f"no drift ({len(live['models'])} models)")
        return 0

    report = format_report(diff)
    print(report, end="")
    webhook = os.environ.get("FEISHU_WEBHOOK_URL", "").strip()
    if not args.no_feishu and webhook:
        try:
            send_feishu(report, webhook)
            print("feishu sent", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"feishu failed: {exc}", file=sys.stderr)
            return 2
    elif not args.no_feishu:
        print("FEISHU_WEBHOOK_URL 未设置，已跳过飞书", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
