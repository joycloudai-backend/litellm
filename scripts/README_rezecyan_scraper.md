# Rezecyan 价格同步

```bash
pip install playwright && playwright install chromium

# 第一步（必须）：登录一次并保存登录态（~/.cache/rezecyan_state.json）
# ali-of-pro 的 group_ratio 与弹窗分组面板只在登录后下发，未登录看到的是 default 分组价
python3 scripts/sync_rezecyan_prices.py --login

# Playwright 打开 https://cn.rezecyan.com/pricing
# 只收录 enable_groups 含 ali-of-pro 的模型；写入源表并同步两份 model_prices JSON（先删旧 rezecyan/* 再写入，去重）
python3 scripts/sync_rezecyan_prices.py --scrape

python3 scripts/sync_rezecyan_prices.py --scrape --dry-run      # 只打印
python3 scripts/sync_rezecyan_prices.py --scrape --sheet-only   # 只写 rezecyan_prices.json
python3 scripts/sync_rezecyan_prices.py --scrape --no-headless  # 调试看浏览器
python3 scripts/sync_rezecyan_prices.py                         # 仅从源表生成 USD 价格
python3 scripts/sync_rezecyan_prices.py --self-test
```

价格只取 `ali-of-pro` 分组：弹窗按 `code.rz-detail-group__key == ali-of-pro` 精确匹配面板，公式换算用 `/api/pricing` 下发的 `group_ratio["ali-of-pro"]`。两者都拿不到（未登录）时 `--scrape` 直接报错，绝不回退 default——default 与 ali-of-pro 价格不同，回退会写入错误价格。分档模型写入全部档位（`*_above_{N}k_tokens`）。

## 监视（开发机，不 Apply）

`watch_rezecyan_pricing.py` 只 `GET /api/pricing`（无需登录），和 `rezecyan_pricing_snapshot.json` 对比。不写源表、不写 LiteLLM 价表、不跑浏览器。

```bash
# 首次：生成快照
python3 scripts/watch_rezecyan_pricing.py --init

# 定时对比（crontab）。有漂移打印报告并 exit 1；配了 FEISHU_WEBHOOK_URL 则发飞书
export FEISHU_WEBHOOK_URL='https://open.feishu.cn/open-apis/bot/v2/hook/...'
python3 scripts/watch_rezecyan_pricing.py

python3 scripts/watch_rezecyan_pricing.py --no-feishu   # 只打印
python3 scripts/watch_rezecyan_pricing.py --self-test
```

盯梢分组写在快照的 `watched_groups`（当前：`default` / `tx-glm53` / `ali-dspro813` / `ali-dsflash731` / `idc-k3-svip` / `ali-glm-svip` / `ali-vip-svip`）。确认漂移后 `--init` 接受新基线。Apply 仍是人审后 `--scrape`。公式仅大小写变化且各档 CNY 不变（如 `tier("base")` → `tier("BASE")`）不告警，脚本会自行回写快照。

开发机 crontab（`run-watch-rezecyan-pricing.sh` 会 source `ops.env`）：

```cron
# 测试：每分钟
* * * * * /data/users/aliang/python/litellm/scripts/run-watch-rezecyan-pricing.sh >> /tmp/reze-watch.log 2>&1

# 通过后改成每 6 小时
0 */6 * * * /data/users/aliang/python/litellm/scripts/run-watch-rezecyan-pricing.sh >> /tmp/reze-watch.log 2>&1
```

`crontab -e` 改间隔；`crontab -r` 停掉。
