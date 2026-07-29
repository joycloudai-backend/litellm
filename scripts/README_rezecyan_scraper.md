# Rezecyan 价格同步

```bash
pip install playwright && playwright install chromium

# 第一步（必须）：登录一次并保存登录态（~/.cache/rezecyan_state.json）
# ali-of-pro 的 group_ratio 与弹窗分组面板只在登录后下发，未登录看到的是 default 分组价
python3 scripts/sync_rezecyan_prices.py --login

# Playwright 打开 https://www.rezecyan.com/pricing
# 只收录 enable_groups 含 ali-of-pro 的模型；写入源表并同步两份 model_prices JSON（先删旧 rezecyan/* 再写入，去重）
python3 scripts/sync_rezecyan_prices.py --scrape

python3 scripts/sync_rezecyan_prices.py --scrape --dry-run      # 只打印
python3 scripts/sync_rezecyan_prices.py --scrape --sheet-only   # 只写 rezecyan_prices.json
python3 scripts/sync_rezecyan_prices.py --scrape --no-headless  # 调试看浏览器
python3 scripts/sync_rezecyan_prices.py                         # 仅从源表生成 USD 价格
python3 scripts/sync_rezecyan_prices.py --self-test
```

价格只取 `ali-of-pro` 分组：弹窗按 `code.rz-detail-group__key == ali-of-pro` 精确匹配面板，公式换算用 `/api/pricing` 下发的 `group_ratio["ali-of-pro"]`。两者都拿不到（未登录）时 `--scrape` 直接报错，绝不回退 default——default 与 ali-of-pro 价格不同，回退会写入错误价格。分档模型写入全部档位（`*_above_{N}k_tokens`）。
