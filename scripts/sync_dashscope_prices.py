#!/usr/bin/env python3
"""Sync Alibaba Cloud Model Studio (DashScope) list prices into model_prices_and_context_window.json.

Product entry (international console, SPA, needs login + JS):
  https://modelstudio.console.alibabacloud.com/{region}?tab=doc#/doc/?type=model&url=prices
  e.g. Singapore: .../ap-southeast-1?tab=doc#/doc/?type=model&url=prices

That console shell cannot be scraped with plain HTTP (returns only the React bootstrap HTML).
Automated fetch therefore uses the public help doc that carries the same pricing tables:
  https://www.alibabacloud.com/help/en/model-studio/model-pricing

Workflow options:
  1) Default --source help: fetch the public doc (recommended, no login).
  2) --html-file PATH: paste/save HTML from the console doc tab (or help page) and parse offline.
  3) --source console: open the console URL via Playwright if installed; otherwise print the
     exact console URL and ask you to Save As → --html-file (supports --page hints for manual
     pagination / "More models" expansion).

Key naming (see joycloud base docs/dashscope方案.md):
  - Qwen / embedding / rerank: dashscope/<model>[<-cn|-hk|-eu>]
  - Third-party: dashscope/<vendor>/<model>[<-cn|-hk|-eu>], litellm_provider=openai
  - Singapore International = main tier (no suffix)

Examples:
  python scripts/sync_dashscope_prices.py --region singapore --models qwen3.7-plus,kimi-k2.7-code --dry-run
  python scripts/sync_dashscope_prices.py --region beijing --models qwen3.7-max
  python scripts/sync_dashscope_prices.py --html-file /tmp/console-prices.html --region singapore
  python scripts/sync_dashscope_prices.py --source console --region singapore --list-models

Notes:
  - Help doc is a single page (region sections + "More models"); --page is informational only.
  - Cache unit prices are on a separate doc and are not scraped (existing cache_* fields kept).
  - Promotional labels are ignored; "List price $X" is preferred when present.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Optional

PRICING_URL = "https://www.alibabacloud.com/help/en/model-studio/model-pricing"
CONSOLE_URL_TMPL = (
    "https://modelstudio.console.alibabacloud.com/{console_region}"
    "?tab=doc#/doc/?type=model&url=prices"
)
SOURCE = PRICING_URL
DEFAULT_JSON = Path(__file__).resolve().parents[1] / "model_prices_and_context_window.json"

# --region → help-doc h4 title + default price-tier suffix + console path
REGION_MAP: dict[str, dict] = {
    "singapore": {
        "headings": ("Singapore",),
        "suffix": "",
        "default_scopes": ("International",),
        "console_region": "ap-southeast-1",
    },
    "beijing": {
        "headings": ("China (Beijing)", "China(Beijing)"),
        "suffix": "-cn",
        "default_scopes": ("Chinese mainland",),
        "console_region": "cn-beijing",
    },
    "hongkong": {
        "headings": ("Hong Kong (China)", "China (Hong Kong)", "China(Hong Kong)"),
        "suffix": "-hk",
        "default_scopes": ("Hong Kong (China)", "Global", "International"),
        "console_region": "cn-hongkong",
    },
    "frankfurt": {
        "headings": ("Germany (Frankfurt)",),
        "suffix": "-eu",
        "default_scopes": ("EU",),
        "console_region": "eu-central-1",
    },
    "virginia": {
        "headings": ("US (Virginia)",),
        "suffix": "",
        "default_scopes": ("US", "Global", "International"),
        "console_region": "us-east-1",
    },
    "tokyo": {
        "headings": ("Japan (Tokyo)",),
        "suffix": "",
        "default_scopes": ("Japan", "Global", "International"),
        "console_region": "ap-northeast-1",
    },
}

VENDOR_PREFIXES = ("kimi", "glm", "deepseek", "minimax")
MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*")
PRICE_LIST_RE = re.compile(r"List\s*price\s*\$([0-9]+(?:\.[0-9]+)?)", re.I)
PRICE_ANY_RE = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)")
RANGE_RE = re.compile(
    r"(?:0\s*<\s*)?Token\s*≤\s*(\d+)\s*([KM])|"
    r"(\d+)\s*([KM])\s*<\s*Token\s*≤\s*(\d+)\s*([KM])|"
    r"(\d+)\s*([KM])\s*<\s*输入\s*≤\s*(\d+)\s*([KM])",
    re.I,
)


def _unit_to_tokens(n: int, unit: str) -> float:
    u = unit.upper()
    if u == "K":
        return float(n * 1000)
    if u == "M":
        return float(n * 1_000_000)
    raise ValueError(unit)


def parse_token_range(text: str) -> Optional[list[float]]:
    s = re.sub(r"\s+", "", text or "")
    if not s or re.search(r"no\s*tiered|flat-rate|flatrate|-", s, re.I):
        return None
    # 0<Token≤256K
    m = re.fullmatch(r"(?:0<)?Token≤(\d+)([KM])", s, re.I)
    if m:
        return [0.0, _unit_to_tokens(int(m.group(1)), m.group(2))]
    # 256K<Token≤1M
    m = re.fullmatch(r"(\d+)([KM])<Token≤(\d+)([KM])", s, re.I)
    if m:
        return [
            _unit_to_tokens(int(m.group(1)), m.group(2)),
            _unit_to_tokens(int(m.group(3)), m.group(4)),
        ]
    return None


def parse_price_usd_per_million(text: str) -> Optional[float]:
    if not text:
        return None
    if re.search(r"discontinued|free|—|–", text, re.I) and "$" not in text:
        return None
    m = PRICE_LIST_RE.search(text)
    if m:
        return float(m.group(1))
    m = PRICE_ANY_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def extract_model_id(cell: str) -> Optional[str]:
    if not cell:
        return None
    s = re.sub(r"\s+", " ", cell.strip())
    # HTML often concatenates <p>qwen3.7-plus</p><blockquote>Currently... without a space
    s = re.sub(
        r"(?i)(?<=[A-Za-z0-9._-])(?=(Currently|context\s*caching|Context\s*Cache|50%\s*batch))",
        " ",
        s,
    )
    s = re.split(
        r"(?i)\bCurrently\b|\bcontext caching\b|\bContext Cache\b|\b50%\b|\bLimited-time\b|\bList price\b",
        s,
        maxsplit=1,
    )[0].strip(" \t-|")
    m = MODEL_ID_RE.match(s)
    if not m:
        return None
    mid = m.group(0)
    # reject token-range debris / header junk
    if re.fullmatch(r"\d+[KMkm]?", mid):
        return None
    if mid.lower() in {"token", "model", "input", "output", "mode", "deployment"}:
        return None
    return mid


def per_token(per_million: Optional[float]) -> Optional[float]:
    if per_million is None:
        return None
    return per_million / 1_000_000.0


def price_key(model_id: str, suffix: str) -> tuple[str, str]:
    """Return (json_key, litellm_provider)."""
    vendor = next((v for v in VENDOR_PREFIXES if model_id.startswith(v)), None)
    if vendor:
        return f"dashscope/{vendor}/{model_id}{suffix}", "openai"
    return f"dashscope/{model_id}{suffix}", "dashscope"


def infer_mode(section_path: str, model_id: str) -> str:
    path = section_path.lower()
    if "rerank" in path or "rerank" in model_id:
        return "rerank"
    if "embedding" in path or "embedding" in model_id:
        return "embedding"
    # "image" as a whole hyphen segment (qwen-image, qwen-image-edit-max) so
    # vision-chat models like qwen-vl-* stay "chat"
    if "image generation" in path or "image" in model_id.split("-"):
        return "image_generation"
    return "chat"


@dataclass
class Tier:
    range: Optional[list[float]]
    input_per_m: float
    output_per_m: float
    output_thinking_per_m: Optional[float] = None


@dataclass
class ModelPrice:
    model_id: str
    region_heading: str
    deployment_scope: str
    section_path: str
    tiers: list[Tier] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return infer_mode(self.section_path, self.model_id)


class HelpDocParser(HTMLParser):
    """Walk help-doc HTML: track headings + table rows."""

    def __init__(self) -> None:
        super().__init__()
        self.in_doc = False
        self.doc_depth = 0
        self.heading_level: Optional[int] = None
        self.heading_buf: list[str] = []
        self.headings: list[tuple[int, str]] = []  # (level, text) stack via path
        self.heading_path: list[tuple[int, str]] = []

        self.in_table = False
        self.in_tr = False
        self.in_cell = False
        self.cell_tag = ""
        self.cell_buf: list[str] = []
        self.row_cells: list[str] = []
        self.table_rows: list[list[str]] = []
        self.current_region: Optional[str] = None
        self.pending_tables: list[tuple[str, str, list[list[str]]]] = []
        # (region_heading, section_path, rows)

    def _path_str(self) -> str:
        return " > ".join(t for _, t in self.heading_path)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_d = dict(attrs)
        classes = (attrs_d.get("class") or "").split()
        if tag == "div" and "icms-help-docs-content" in classes:
            self.in_doc = True
            self.doc_depth = 1
            return
        if self.in_doc and tag == "div":
            self.doc_depth += 1

        if not self.in_doc:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.heading_level = int(tag[1])
            self.heading_buf = []
            return

        if tag == "table":
            self.in_table = True
            self.table_rows = []
            return

        if self.in_table and tag == "tr":
            self.in_tr = True
            self.row_cells = []
            return

        if self.in_tr and tag in ("td", "th"):
            self.in_cell = True
            self.cell_tag = tag
            self.cell_buf = []
            return
        # insert whitespace between inline blocks so "plusCurrently" becomes separable
        if self.in_cell and tag in ("p", "div", "blockquote", "br", "li"):
            self.cell_buf.append(" ")
            return

    def handle_endtag(self, tag: str) -> None:
        if not self.in_doc and tag != "div":
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self.heading_level:
            text = re.sub(r"\s+", " ", "".join(self.heading_buf)).strip()
            level = self.heading_level
            self.heading_level = None
            if text:
                while self.heading_path and self.heading_path[-1][0] >= level:
                    self.heading_path.pop()
                self.heading_path.append((level, text))
                if level == 4:
                    self.current_region = text
            return

        if tag == "table" and self.in_table:
            self.in_table = False
            if self.current_region and self.table_rows:
                self.pending_tables.append(
                    (self.current_region, self._path_str(), self.table_rows)
                )
            self.table_rows = []
            return

        if tag == "tr" and self.in_tr:
            self.in_tr = False
            if self.row_cells:
                self.table_rows.append(self.row_cells)
            return

        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            cell = re.sub(r"\s+", " ", "".join(self.cell_buf)).strip()
            self.row_cells.append(cell)
            return

        if tag == "div" and self.in_doc:
            self.doc_depth -= 1
            if self.doc_depth <= 0:
                self.in_doc = False

    def handle_data(self, data: str) -> None:
        if self.heading_level is not None:
            self.heading_buf.append(data)
        elif self.in_cell:
            self.cell_buf.append(data)


def _col_index(header: list[str], *needles: str) -> Optional[int]:
    lowered = [h.lower() for h in header]
    for i, h in enumerate(lowered):
        if all(n.lower() in h for n in needles):
            return i
    for i, h in enumerate(lowered):
        if any(n.lower() == h or n.lower() in h for n in needles):
            return i
    return None


def tables_to_models(
    tables: list[tuple[str, str, list[list[str]]]],
    region_headings: Iterable[str],
) -> list[ModelPrice]:
    wanted = {h.lower() for h in region_headings}
    out: list[ModelPrice] = []
    for region, section_path, rows in tables:
        if region.lower() not in wanted:
            continue
        if not rows:
            continue
        # find header row (contains "Model")
        header_i = next(
            (i for i, r in enumerate(rows) if any("model" in c.lower() for c in r)),
            None,
        )
        if header_i is None:
            continue
        header = rows[header_i]
        # skip sub-header rows like "Non-Thinking mode | Thinking mode"
        data_rows = rows[header_i + 1 :]

        i_model = _col_index(header, "model")
        i_scope = _col_index(header, "deployment", "scope") or _col_index(header, "scope")
        i_tokens = _col_index(header, "input", "token") or _col_index(header, "token")
        i_in = _col_index(header, "input", "price")
        # first output price column
        i_out = None
        i_out_think = None
        for i, h in enumerate(header):
            hl = h.lower()
            if "output" in hl and "price" in hl:
                if i_out is None:
                    i_out = i
                elif i_out_think is None and ("thinking" in hl or "chain" in hl):
                    i_out_think = i
        # embedding tables: only input price
        if i_in is None:
            i_in = _col_index(header, "price")
        if i_model is None or i_in is None:
            continue

        current: Optional[ModelPrice] = None
        for row in data_rows:
            if not row or all(not c.strip() for c in row):
                continue
            # pad
            while len(row) < len(header):
                row.append("")

            mid_raw = row[i_model] if i_model < len(row) else ""
            mid = extract_model_id(mid_raw) if mid_raw.strip() else None
            scope = (row[i_scope] if i_scope is not None and i_scope < len(row) else "") or ""
            tok = row[i_tokens] if i_tokens is not None and i_tokens < len(row) else ""
            inp = parse_price_usd_per_million(row[i_in] if i_in < len(row) else "")
            out_p = (
                parse_price_usd_per_million(row[i_out] if i_out is not None and i_out < len(row) else "")
                if i_out is not None
                else 0.0
            )
            think_p = (
                parse_price_usd_per_million(
                    row[i_out_think] if i_out_think is not None and i_out_think < len(row) else ""
                )
                if i_out_think is not None
                else None
            )

            # continuation tier row (empty model id)
            if mid is None:
                if current is None or inp is None:
                    continue
                if out_p is None:
                    out_p = current.tiers[-1].output_per_m if current.tiers else 0.0
                current.tiers.append(
                    Tier(
                        range=parse_token_range(tok),
                        input_per_m=inp,
                        output_per_m=out_p or 0.0,
                        output_thinking_per_m=think_p,
                    )
                )
                continue

            if inp is None:
                continue
            if out_p is None:
                out_p = 0.0

            current = ModelPrice(
                model_id=mid,
                region_heading=region,
                deployment_scope=scope.strip(),
                section_path=section_path,
                tiers=[
                    Tier(
                        range=parse_token_range(tok),
                        input_per_m=inp,
                        output_per_m=out_p,
                        output_thinking_per_m=think_p,
                    )
                ],
            )
            out.append(current)
    return out


def fetch_html(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; litellm-dashscope-price-sync/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_help_html(html: str) -> list[tuple[str, str, list[list[str]]]]:
    parser = HelpDocParser()
    parser.feed(html)
    return parser.pending_tables


def filter_models(
    models: list[ModelPrice],
    model_names: Optional[set[str]],
    scopes: Optional[set[str]],
) -> list[ModelPrice]:
    out = []
    for m in models:
        if model_names and m.model_id not in model_names:
            continue
        if scopes:
            # empty scope column (some embedding tables) → keep
            if m.deployment_scope and m.deployment_scope not in scopes:
                # allow case-insensitive / substring for "Hong Kong (China)"
                if not any(
                    s.lower() == m.deployment_scope.lower()
                    or s.lower() in m.deployment_scope.lower()
                    or m.deployment_scope.lower() in s.lower()
                    for s in scopes
                ):
                    continue
        out.append(m)
    return out


def build_entry(model: ModelPrice, provider: str) -> dict:
    mode = model.mode
    tiers = model.tiers
    use_tiered = len(tiers) > 1 and all(t.range is not None for t in tiers)

    entry: dict = {
        "litellm_provider": provider,
        "mode": mode,
        "provider_pricing_currency": "USD",
        "source": SOURCE,
    }

    if mode == "chat":
        entry["supports_function_calling"] = True
        entry["supports_reasoning"] = True
        entry["supports_tool_choice"] = True

    if mode == "image_generation":
        # DashScope image models have a single flat price (billed on output
        # image tokens only), so no input_cost_per_token
        entry["supported_endpoints"] = ["/v1/images/generations"]
        entry["output_cost_per_token"] = per_token(tiers[0].output_per_m)
        return entry

    if mode in ("embedding", "rerank"):
        entry["input_cost_per_token"] = per_token(tiers[0].input_per_m)
        entry["output_cost_per_token"] = 0.0
        return entry

    if use_tiered:
        tiered = []
        for t in tiers:
            item = {
                "input_cost_per_token": per_token(t.input_per_m),
                "output_cost_per_token": per_token(t.output_per_m),
                "range": list(t.range) if t.range else [0.0, 0.0],
            }
            if (
                t.output_thinking_per_m is not None
                and abs(t.output_thinking_per_m - t.output_per_m) > 1e-9
            ):
                item["output_cost_per_reasoning_token"] = per_token(t.output_thinking_per_m)
            tiered.append(item)
        entry["tiered_pricing"] = tiered
        entry["input_cost_per_token"] = tiered[0]["input_cost_per_token"]
        entry["output_cost_per_token"] = tiered[0]["output_cost_per_token"]
        if "output_cost_per_reasoning_token" in tiered[0]:
            entry["output_cost_per_reasoning_token"] = tiered[0]["output_cost_per_reasoning_token"]
        return entry

    t = tiers[0]
    entry["input_cost_per_token"] = per_token(t.input_per_m)
    entry["output_cost_per_token"] = per_token(t.output_per_m)
    if (
        t.output_thinking_per_m is not None
        and abs(t.output_thinking_per_m - t.output_per_m) > 1e-9
    ):
        entry["output_cost_per_reasoning_token"] = per_token(t.output_thinking_per_m)
    return entry


PRICE_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "output_cost_per_reasoning_token",
    "tiered_pricing",
    "provider_pricing_currency",
    "source",
    "litellm_provider",
    "mode",
)


def merge_entry(existing: dict, new: dict) -> dict:
    """Update price-related fields; keep max_tokens / supports_* / cache_* etc."""
    merged = dict(existing)
    for k in PRICE_FIELDS:
        if k in new:
            merged[k] = new[k]
    # image models bill a single flat price; drop a stale input cost left over
    # from when they were mis-synced as chat models
    if new.get("mode") == "image_generation" and "input_cost_per_token" not in new:
        merged.pop("input_cost_per_token", None)
    # ensure capability flags exist for new chat models without wiping customs
    for k in ("supports_function_calling", "supports_reasoning", "supports_tool_choice"):
        if k in new and k not in merged:
            merged[k] = new[k]
    return merged


# Top-level entries are `    "key": {` at indent 4; nested objects close at
# indent >= 8, so "\n    }" uniquely terminates a top-level block.
TOP_BLOCK_RE = re.compile(r'^    "([^"]+)": \{', re.M)


def find_top_blocks(text: str) -> dict[str, tuple[int, int]]:
    """Map each top-level key to the (start, end) span of its first block."""
    blocks: dict[str, tuple[int, int]] = {}
    for m in TOP_BLOCK_RE.finditer(text):
        end = text.index("\n    }", m.end()) + len("\n    }")
        blocks.setdefault(m.group(1), (m.start(), end))
    return blocks


def render_block(key: str, entry: dict) -> str:
    body = json.dumps(entry, indent=4, ensure_ascii=False).replace("\n", "\n    ")
    return f'    "{key}": {body}'


def apply_updates(text: str, updates: dict[str, dict]) -> tuple[str, list[str], list[str]]:
    """Splice updates into the raw JSON text, touching only the affected blocks.

    A whole-file json.loads -> json.dumps round trip is unsafe here: the upstream
    file contains duplicate top-level keys (dict load silently collapses them)
    and any re-serialization rewrites unrelated providers' entries.
    """
    blocks = find_top_blocks(text)
    dash_sorted = sorted(k for k in blocks if k.startswith("dashscope/"))
    added, updated = [], []
    edits: list[tuple[int, int, str]] = []
    inserts: dict[int, list[str]] = defaultdict(list)

    for key, entry in sorted(updates.items()):
        if key in blocks:
            start, end = blocks[key]
            prefix = f'    "{key}": '
            existing = json.loads(text[start + len(prefix) : end])
            edits.append((start, end, render_block(key, merge_entry(existing, entry))))
            updated.append(key)
            continue
        if not dash_sorted:
            raise RuntimeError("no existing dashscope/* block to anchor insertion")
        successor = next((k for k in dash_sorted if k > key), None)
        if successor is not None:
            pos = blocks[successor][0]
        else:
            last_end = max(blocks[k][1] for k in dash_sorted)
            # ponytail: assumes the last dashscope block is not the file's final
            # entry (always true here); otherwise comma handling would differ
            if text[last_end] != ",":
                raise RuntimeError("last dashscope block is the final entry; cannot append")
            pos = last_end + 2
        inserts[pos].append(render_block(key, entry))
        added.append(key)

    for pos, rendered in inserts.items():
        edits.append((pos, pos, "".join(b + ",\n" for b in rendered)))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    json.loads(text)  # sanity: result must still be valid JSON
    return text, added, updated


def console_url(region: str) -> str:
    return CONSOLE_URL_TMPL.format(console_region=REGION_MAP[region]["console_region"])


def print_console_manual_instructions(region: str, page: Optional[int]) -> None:
    url = console_url(region)
    print(
        "Console pricing UI is a login-walled SPA and cannot be fetched with plain HTTP.\n"
        f"  Console URL: {url}\n"
        "Manual steps:\n"
        "  1. Open the URL above, log in, wait until the pricing tables render.\n"
        "  2. If the table is paginated or has 'More models', expand/flip to the page you need"
        + (f" (you asked for page={page})." if page is not None else ".")
        + "\n"
        "  3. Browser Save As → Webpage, Complete (or copy the article HTML).\n"
        "  4. Re-run with: --html-file /path/to/saved.html --region "
        f"{region}\n"
        "Or use the public mirror (same tables, no login):\n"
        f"  --source help  (fetches {PRICING_URL})",
        file=sys.stderr,
    )


def fetch_console_html_playwright(region: str, page: Optional[int], timeout_ms: int = 90000) -> str:
    """Optional path: render console doc tab with Playwright (must be installed + logged-in storage)."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "playwright not installed. pip install playwright && playwright install chromium"
        ) from e

    url = console_url(region)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        # Optional: reuse a logged-in session exported by the user
        storage = Path.home() / ".cache" / "dashscope_console_state.json"
        if storage.exists():
            context = browser.new_context(storage_state=str(storage))
        page_obj = context.new_page()
        page_obj.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # Wait for any pricing table / model id text
        try:
            page_obj.wait_for_selector("table, .markdown-body, .icms-help-docs-content", timeout=timeout_ms)
        except Exception:
            pass
        if page is not None:
            # Best-effort: click pagination / "More models" if present
            for label in (f"{page}", "下一页", "Next", "More models", "更多模型"):
                loc = page_obj.get_by_text(label, exact=False)
                if loc.count():
                    try:
                        loc.first.click(timeout=3000)
                        page_obj.wait_for_timeout(1500)
                    except Exception:
                        pass
        html = page_obj.content()
        browser.close()
    if "qwen" not in html.lower() and "Model ID" not in html:
        raise RuntimeError(
            "console page rendered but no pricing tables found "
            "(likely not logged in). Export storage_state after manual login:\n"
            "  playwright codegen "
            f"'{url}'\n"
            f"then save storage to {storage}"
        )
    return html


def load_html(args: argparse.Namespace) -> str:
    """Resolve HTML from --html-file / --source help|console."""
    if args.html_file:
        print(f"loaded HTML from {args.html_file}", file=sys.stderr)
        return args.html_file.read_text(encoding="utf-8", errors="replace")

    source = args.source
    if source == "console":
        print(f"console URL: {console_url(args.region)}", file=sys.stderr)
        try:
            return fetch_console_html_playwright(args.region, args.page)
        except Exception as e:
            print(f"console fetch failed: {e}", file=sys.stderr)
            print_console_manual_instructions(args.region, args.page)
            raise SystemExit(1) from e

    # default: help
    url = args.url
    print(
        f"fetching public help doc (console SPA mirror): {url}\n"
        f"  console entry: {console_url(args.region)}",
        file=sys.stderr,
    )
    try:
        return fetch_html(url)
    except Exception as e:
        print(
            f"fetch failed: {e}\n"
            "Fallback: open the console page, Save As HTML, then --html-file ...",
            file=sys.stderr,
        )
        print_console_manual_instructions(args.region, args.page)
        raise SystemExit(1) from e


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--region",
        required=True,
        choices=sorted(REGION_MAP.keys()),
        help="Geographic section on the pricing page (maps to price-tier suffix + console path)",
    )
    p.add_argument(
        "--models",
        default="",
        help="Comma-separated model IDs to sync (default: all models in region/scope)",
    )
    p.add_argument(
        "--scope",
        default="",
        help="Filter deployment scope (comma-separated). Default: region-specific "
        "(e.g. singapore→International, beijing→Chinese mainland, frankfurt→EU)",
    )
    p.add_argument(
        "--suffix",
        default=None,
        help="Override price-tier suffix (default from --region: ''|-cn|-hk|-eu)",
    )
    p.add_argument("--json-path", type=Path, default=DEFAULT_JSON)
    p.add_argument(
        "--source",
        choices=("help", "console"),
        default="help",
        help="help=public doc (default); console=Playwright against modelstudio.console.alibabacloud.com",
    )
    p.add_argument("--url", default=PRICING_URL, help="Override help-doc URL when --source help")
    p.add_argument("--html-file", type=Path, help="Parse a saved console/help HTML file (skips fetch)")
    p.add_argument(
        "--page",
        type=int,
        default=None,
        help="Hint for console pagination / More models (Playwright best-effort, or manual Save As)",
    )
    p.add_argument("--list-models", action="store_true", help="List parsed model IDs and exit")
    p.add_argument("--dry-run", action="store_true", help="Print actions without writing JSON")
    p.add_argument(
        "--all-scopes",
        action="store_true",
        help="Do not filter by deployment scope (take every row under the region heading)",
    )
    p.add_argument(
        "--print-console-url",
        action="store_true",
        help="Print the international console pricing URL for --region and exit",
    )
    args = p.parse_args(argv)

    if args.print_console_url:
        print(console_url(args.region))
        return 0

    if args.page is not None and args.source == "help" and not args.html_file:
        print(
            f"note: --page={args.page} only applies to --source console or manual --html-file; "
            "the public help doc is a single page (region sections + More models).",
            file=sys.stderr,
        )

    region_cfg = REGION_MAP[args.region]
    suffix = region_cfg["suffix"] if args.suffix is None else args.suffix

    html = load_html(args)

    tables = parse_help_html(html)
    if not tables:
        print(
            "no tables parsed — if this was a console Save As of the SPA shell only, "
            "wait until tables fully render, or use --source help.",
            file=sys.stderr,
        )
        print_console_manual_instructions(args.region, args.page)
        return 1

    models = tables_to_models(tables, region_cfg["headings"])
    if not models:
        print(
            f"no models under region headings {region_cfg['headings']}. "
            f"Available region headings sample: "
            f"{sorted({t[0] for t in tables})[:20]}",
            file=sys.stderr,
        )
        return 1

    model_filter = {m.strip() for m in args.models.split(",") if m.strip()} or None
    if args.all_scopes:
        scope_filter = None
    elif args.scope:
        scope_filter = {s.strip() for s in args.scope.split(",") if s.strip()}
    else:
        scope_filter = set(region_cfg["default_scopes"])

    filtered = filter_models(models, model_filter, scope_filter)

    if model_filter:
        found = {m.model_id for m in filtered}
        missing = sorted(model_filter - found)
        if missing:
            available = sorted({m.model_id for m in models})
            print(f"models not found in region/scope: {missing}", file=sys.stderr)
            print(f"available under region (any scope): {available[:50]}", file=sys.stderr)
            if sys.stdin.isatty():
                ans = input("Continue with found models only? [y/N] ").strip().lower()
                if ans not in ("y", "yes"):
                    return 2
            else:
                return 2

    if args.list_models:
        by_id: dict[str, list[ModelPrice]] = defaultdict(list)
        for m in filtered:
            by_id[m.model_id].append(m)
        for mid in sorted(by_id):
            scopes = sorted({x.deployment_scope or "-" for x in by_id[mid]})
            key, prov = price_key(mid, suffix)
            print(f"{mid}\tscopes={scopes}\tkey={key}\tprovider={prov}")
        print(f"# {len(by_id)} models", file=sys.stderr)
        return 0

    if not filtered:
        print("nothing to update after filters", file=sys.stderr)
        return 1

    # collapse duplicate model_id+scope: last wins; prefer more tiers
    chosen: dict[str, ModelPrice] = {}
    for m in filtered:
        prev = chosen.get(m.model_id)
        if prev is None or len(m.tiers) >= len(prev.tiers):
            chosen[m.model_id] = m

    updates: dict[str, dict] = {}
    for mid, m in sorted(chosen.items()):
        key, provider = price_key(mid, suffix)
        updates[key] = build_entry(m, provider)
        print(
            f"plan {key}: {m.deployment_scope or '-'} "
            f"tiers={len(m.tiers)} in={m.tiers[0].input_per_m}/M out={m.tiers[0].output_per_m}/M",
            file=sys.stderr,
        )

    if args.region == "tokyo" and suffix == "":
        print(
            "warning: tokyo defaults to main-tier keys (no suffix). "
            "Global prices often differ from Singapore International — "
            "use --dry-run and/or --suffix to avoid overwriting main-tier rates.",
            file=sys.stderr,
        )

    if args.dry_run:
        print(f"dry-run: would touch {len(updates)} keys", file=sys.stderr)
        return 0

    raw = args.json_path.read_text(encoding="utf-8")
    new_text, added, updated = apply_updates(raw, updates)
    args.json_path.write_text(new_text, encoding="utf-8")
    print(f"wrote {args.json_path}: added={len(added)} updated={len(updated)}", file=sys.stderr)
    for k in added:
        print(f"  + {k}")
    for k in updated:
        print(f"  ~ {k}")
    return 0


# --- minimal self-check (stdlib assert) ---
def _self_check() -> None:
    assert extract_model_id("qwen3.7-plus Currently equivalent to x") == "qwen3.7-plus"
    assert parse_price_usd_per_million("List price $2.5 Limited-time 50% off") == 2.5
    assert parse_price_usd_per_million("$0.276") == 0.276
    assert parse_token_range("0<Token≤256K") == [0.0, 256000.0]
    assert parse_token_range("256K<Token≤1M") == [256000.0, 1_000_000.0]
    assert price_key("qwen3.7-plus", "-cn") == ("dashscope/qwen3.7-plus-cn", "dashscope")
    assert price_key("kimi-k2.5", "-cn") == ("dashscope/kimi/kimi-k2.5-cn", "openai")
    assert infer_mode("Text generation", "qwen3.7-plus") == "chat"
    assert infer_mode("Image generation", "qwen-image-max") == "image_generation"
    assert infer_mode("More models", "qwen-image-edit-max-2026-01-16") == "image_generation"
    assert infer_mode("Visual understanding", "qwen-vl-plus") == "chat"
    assert infer_mode("Text embedding", "text-embedding-v4") == "embedding"
    img = build_entry(
        ModelPrice(
            model_id="qwen-image-max",
            region_heading="Singapore",
            deployment_scope="International",
            section_path="Image generation",
            tiers=[Tier(range=None, input_per_m=0.075, output_per_m=0.075)],
        ),
        "dashscope",
    )
    assert img["mode"] == "image_generation"
    assert "input_cost_per_token" not in img, "image models bill a single flat price"
    assert img["output_cost_per_token"] == 7.5e-08
    assert img["supported_endpoints"] == ["/v1/images/generations"]
    stale = {"mode": "image_generation", "input_cost_per_token": 1e-08, "output_cost_per_token": 1e-08}
    assert "input_cost_per_token" not in merge_entry(stale, img), "merge must drop stale input cost"
    assert "ap-southeast-1" in console_url("singapore")
    assert "url=prices" in console_url("singapore")
    # regression: duplicate top-level keys and unrelated providers must survive
    # the write untouched (byte-for-byte outside the dashscope blocks)
    raw = (
        "{\n"
        '    "bedrock/dup": {\n        "a": 1\n    },\n'
        '    "dashscope/qwen-b": {\n        "input_cost_per_token": 1e-06,\n        "max_tokens": 8\n    },\n'
        '    "bedrock/dup": {\n        "a": 2\n    },\n'
        '    "zz/tail": {\n        "a": 3\n    }\n'
        "}\n"
    )
    out, added2, updated2 = apply_updates(
        raw,
        {
            "dashscope/qwen-a": {"litellm_provider": "dashscope", "input_cost_per_token": 2e-06},
            "dashscope/qwen-b": {"input_cost_per_token": 3e-06},
            "dashscope/qwen-z": {"litellm_provider": "dashscope"},
        },
    )
    assert added2 == ["dashscope/qwen-a", "dashscope/qwen-z"], added2
    assert updated2 == ["dashscope/qwen-b"], updated2
    assert out.count('"bedrock/dup"') == 2, "duplicate keys were collapsed"
    assert '"a": 1' in out and '"a": 2' in out and '"a": 3' in out
    assert '"max_tokens": 8' in out, "merge_entry must keep non-price fields"
    assert '"input_cost_per_token": 3e-06' in out
    assert out.index('"dashscope/qwen-a"') < out.index('"dashscope/qwen-b"') < out.index('"dashscope/qwen-z"')
    print("self-check ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
        raise SystemExit(0)
    raise SystemExit(main())
