# 视频预占预算（硬防超花）E2E 测试方案

> 源码：`litellm/proxy/spend_tracking/budget_reservation.py`、`litellm/proxy/spend_tracking/volcengine_video_billing.py`
> 日期：2026-07-29
> 范围：BytePlus / Volcengine（Seedance）视频 create 的 admission 预占、异步结账补差、失败退预占

## 0. 背景与验收目标

改动前：视频 create 估费返回 `None`，且 create 强制 `response_cost=0`，预占立刻释放；并发多条 create 可在异步计费落地前一起越过 `team.max_budget`。

改动后：

1. **admission 估费**：create 前按分辨率 × fps × 时长（及是否参考视频）算保守上限 USD，写入 budget reservation
2. **挂账**：create 成功返回 `reserved_cost`，`LiteLLM_VideoTaskTable.spend` 先记 provisional，Redis hold 不立刻清零
3. **结账**：任务完成按 `final - provisional` 补差（可负）；失败 / no_charge 退回全部 provisional
4. **硬挡并发**：余额只够 1 条时，第 2 条并发 create 在 admission 被 `BudgetExceededError`（HTTP 429）挡住

本方案以 **真实 LiteLLM proxy + 真实视频 provider** 为主验收（会花真实费用）；单元测试仅作回归兜底。

---

## 1. 环境准备

| 项 | 建议值 | 说明 |
|---|---|---|
| litellm | `http://127.0.0.1:4000` | 本地或 `kubectl port-forward svc/litellm-svc 4000:4000 -n <ns>` |
| Master Key | `$LITELLM_MASTER_KEY` | 管理接口（改 team `max_budget`、查 spend） |
| 租户 Virtual Key | `$VIDEO_KEY` | 已绑定可调用视频模型的 team key |
| 视频模型 | `$VIDEO_MODEL` | 部署的 publish 名，如 `seedance-2.0` / `doubao-seedance-2.0` |
| 价格表 key | `volcengine/doubao-seedance-2.0` 或 `byteplus/dreamina-seedance-2.0` | `model_info.base_model` / `provider_pricing_model` 须能命中带 `volcengine_video_output_cost_per_million_tokens_*` 的条目 |
| 汇率 | 默认 CNY/USD=`7.2` | 可用环境变量 `LITELLM_VOLCENGINE_VIDEO_CNY_PER_USD` 覆盖 |

健康检查：

```bash
curl -s http://127.0.0.1:4000/health/liveliness
# "I'm alive!"
```

前置确认（缺一不可）：

- [ ] 配置未关闭预算预占（不要设 `disable_budget_reservation: true`）
- [ ] team 已设置 `max_budget`（ERP 余额同步后的值，或下方用 admin API 临时压低）
- [ ] 部署模型的 `mode=video_generation`，且价格表有视频单价字段
- [ ] Redis / DualCache 正常（预占计数依赖它）

查 team / 当前 spend：

```bash
# 记下 team_id、max_budget、spend
curl -s "http://127.0.0.1:4000/team/info?team_id=<TEAM_ID>" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | python3 -m json.tool
```

临时压低预算（测完务必改回）：

```bash
curl -s -X POST "http://127.0.0.1:4000/team/update" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"team_id":"<TEAM_ID>","max_budget": <NEW_MAX>}' | python3 -m json.tool
```

---

## 2. 估费公式（验收前先算）

Seedance 预占（保守）：

```text
tokens ≈ W × H × fps × billable_seconds / 1024
provider_spend = unit_price_per_M_tokens × tokens / 1e6
usd = provider_spend / CNY_PER_USD   # BytePlus 价格表多为 USD，则不再除汇率
```

常用分辨率像素（编码尺寸，略大于标称）：

| resolution | W × H |
|---|---|
| 480p | 864 × 496 |
| 720p | 1248 × 704 |
| 1080p | 1920 × 1088 |
| 4k | 3840 × 2160 |

默认：`fps=24`；未传时长按 **12s**；有参考视频时 `billable_seconds += 15`（输入时长未知，取上限）。

示例：Volcengine `doubao-seedance-2.0`、720p、11s、无参考视频、单价 ¥46/M tokens、汇率 7.2：

```bash
python3 - <<'PY'
W,H,fps,sec,price,fx = 1248,704,24,11,46.0,7.2
tokens = W*H*fps*sec/1024
usd = tokens * price / 1_000_000 / fx
print(f"tokens={tokens:.2f} reserved_usd≈{usd:.6f}")
PY
```

记下输出为 `$RESERVE_USD`，后续用 `max_budget ≈ spend + RESERVE_USD` 做临界预算。

有参考视频时同一参数会显著更高（多加最多 15s），并发挡板应用更高估值。

---

## 3. 用例清单

### 3.1 单请求：admission 放行 + provisional 挂账

**目的**：create 立刻占住预算，而不是等到视频完成。

```bash
# 建议 remaining = max_budget - spend 明显大于 RESERVE_USD
CREATE=$(curl -s -w "\nHTTP:%{http_code}" -X POST "http://127.0.0.1:4000/v1/videos" \
  -H "Authorization: Bearer $VIDEO_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$VIDEO_MODEL\",
    \"prompt\": \"一匹马在草原上奔驰\",
    \"seconds\": 5,
    \"resolution\": \"720p\"
  }")
echo "$CREATE"
VIDEO_ID=$(echo "$CREATE" | sed '/^HTTP:/d' | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))")
echo "VIDEO_ID=$VIDEO_ID"
```

**验收**：

- [ ] HTTP 200，返回 `id`（queued/in_progress）
- [ ] create 后立刻查 team：`spend`（或 spend counters）上升约 `$RESERVE_USD`（允许与最终账单有偏差）
- [ ] DB `LiteLLM_VideoTaskTable` 对应行：`billing_state=pending`，`spend ≈ reserved`
- [ ] Spend Logs 若已写 create 行：`spend ≈ reserved`（不是 0）

查状态（可顺便触发 reconcile）：

```bash
curl -s "http://127.0.0.1:4000/v1/videos/$VIDEO_ID" \
  -H "Authorization: Bearer $VIDEO_KEY" | python3 -m json.tool
```

轮询至 `completed` 后：

- [ ] task.`billing_state=billed`，`spend=final`
- [ ] team 最终 spend ≈ 初始 spend + final（不是 2× reserved；差额靠 delta 结清）
- [ ] 若 final < reserved，team spend 应回落（负 delta）；若 final > reserved，应再扣差额

---

### 3.2 余额不足：单条 create 直接 429

**目的**：估费已生效，不再“先建任务后爆账”。

```bash
# 设 remaining < RESERVE_USD（例如 max_budget = current_spend + RESERVE_USD * 0.5）
curl -s -w "\nHTTP:%{http_code}\n" -X POST "http://127.0.0.1:4000/v1/videos" \
  -H "Authorization: Bearer $VIDEO_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$VIDEO_MODEL\",
    \"prompt\": \"一块欧米茄海马150黑盘手表\",
    \"seconds\": 11,
    \"resolution\": \"720p\"
  }"
```

**验收**：

- [ ] HTTP **429**（或业务体含 budget exceeded / `BudgetExceededError`）
- [ ] **没有**新的 `video_id` / pending task
- [ ] team `spend` 不变

---

### 3.3 并发硬防：只够 1 条时第 2 条被挡（核心）

**目的**：验证改动前的超花漏洞已封。

准备：

1. 固定请求体（同模型、同 `seconds`/`resolution`），算出 `$RESERVE_USD`
2. 设 `max_budget = current_spend + RESERVE_USD`（刚好够 1 条；或任意 `remaining < 2×RESERVE` 也应只放行 1 条——视频**禁止** shrink-to-remaining）
3. 同时打 2 个 create

```bash
BODY=$(cat <<EOF
{"model":"$VIDEO_MODEL","prompt":"一匹马在草原上奔驰","seconds":11,"resolution":"720p"}
EOF
)

# 并行发起
curl -s -o /tmp/v1.json -w "%{http_code}" -X POST "http://127.0.0.1:4000/v1/videos" \
  -H "Authorization: Bearer $VIDEO_KEY" -H "Content-Type: application/json" -d "$BODY" &
PID1=$!
curl -s -o /tmp/v2.json -w "%{http_code}" -X POST "http://127.0.0.1:4000/v1/videos" \
  -H "Authorization: Bearer $VIDEO_KEY" -H "Content-Type: application/json" -d "$BODY" &
PID2=$!
wait $PID1; C1=$?
wait $PID2; C2=$?
echo "exit codes: $C1 $C2"
echo "=== resp1 ==="; cat /tmp/v1.json; echo
echo "=== resp2 ==="; cat /tmp/v2.json; echo
```

更稳妥可用 HTTP code 文件：

```bash
curl -s -o /tmp/v1.json -w "%{http_code}" -X POST "http://127.0.0.1:4000/v1/videos" \
  -H "Authorization: Bearer $VIDEO_KEY" -H "Content-Type: application/json" -d "$BODY" > /tmp/c1.txt &
curl -s -o /tmp/v2.json -w "%{http_code}" -X POST "http://127.0.0.1:4000/v1/videos" \
  -H "Authorization: Bearer $VIDEO_KEY" -H "Content-Type: application/json" -d "$BODY" > /tmp/c2.txt &
wait
echo "HTTP: $(cat /tmp/c1.txt) $(cat /tmp/c2.txt)"
```

**验收**：

- [ ] 恰好 **1** 个 HTTP 200 + video id
- [ ] 另 **1** 个 HTTP **429** / BudgetExceeded
- [ ] 仅 1 条 pending/billed video task；不会出现两条都 queued
- [ ] team 占用 ≈ 1 × reserved（完成后再按 final 结算）

失败判定（改动前典型症状）：两条都 200，异步完成后 team spend > max_budget。

---

### 3.4 失败 / no_charge：退回预占

**目的**：provider 失败或最终不计费时，预占必须退回，不能永久占坑。

可选触发方式（按环境选一种）：

- 用非法/极短不可生成参数诱导 `failed`（若 provider 仍接单，改为在 UI/后台取消任务）
- 或对已 create 的任务等待其进入 failed，再 `GET /v1/videos/{id}` 触发 reconcile

**验收**：

- [ ] task.`billing_state=no_charge`（或等价终态），`spend=0`
- [ ] team spend 回到 create 前水平（退回 provisional）
- [ ] 退回后，用同样临界预算再 create **可以**再放行 1 条

---

### 3.5 参考视频：估费抬高且仍能挡并发

带 `input_reference` / content 里含 video URL 的请求，预占应明显高于纯文生视频（billable 多加最多 15s）。

```bash
# 示例：multipart 上传参考视频（字段名以实际 proxy 支持为准）
curl -s -w "\nHTTP:%{http_code}\n" -X POST "http://127.0.0.1:4000/v1/videos" \
  -H "Authorization: Bearer $VIDEO_KEY" \
  -F "model=$VIDEO_MODEL" \
  -F "prompt=with reference video reservation" \
  -F "seconds=5" \
  -F "resolution=720p" \
  -F "input_reference=@/path/to/short.mp4"
```

**验收**：

- [ ] 在「只够纯文生 1 条、不够带参考视频 1 条」的 max_budget 下，带参考视频的 create 被 429
- [ ] 纯文生仍可 200（对照证明估费路径区分了 has_input_video）

---

### 3.6（可选）跨模态：视频预占挡住后续 chat

1. 压低预算到只够 1 条短视频预占
2. create 视频占住
3. 立刻用同一 key 打 `/v1/chat/completions`

**验收**：

- [ ] chat 在视频未完成/未退预占前也可能 429（说明 hold 计入 team 有效占用）
- [ ] 视频结账或失败退回后，chat 恢复（在仍有余额时）

---

## 4. 观测与排查

| 信号 | 预期 |
|---|---|
| create 响应 | 200 + `id`；失败为 429 而非 200 空任务 |
| `GET /team/info` | create 后 spend/占用立刻上升 |
| `LiteLLM_VideoTaskTable` | create：`pending` + provisional spend；完成：`billed` + final；失败：`no_charge` + spend 0 |
| Spend Logs | create 非 0；finalize 后与 final 一致 |
| Proxy log | 可见 reservation / `BudgetExceeded`；billing hook 异常时应有 error，且不应静默双计 |

常见坑：

1. **shrink-to-remaining（已修）**：旧逻辑在 `spend=32.68 max=32.7`（剩 `$0.02`）时会把视频预占缩到 `$0.02` 仍放行，结账必超支。视频/图片 create 现已**硬拒**（`estimate > remaining` → 429）；chat 仍可缩额
2. **临界验收**：`max_budget = spend + 0.02`（远小于 `$RESERVE_USD`）时 create 必须 429；status `GET /v1/videos/{id}` 不应因预算预占失败
3. **价格表未命中**：估费返回 `None` → 退回改动前行为（无预占）；检查 `base_model` / `provider_pricing_model`
4. **汇率不一致**：Volcengine CNY 条目受 `LITELLM_VOLCENGINE_VIDEO_CNY_PER_USD` 影响
5. **ERP 稍后同步 max_budget**：E2E 期间用 LiteLLM admin 直接改 team，避免余额同步把预算改回去
6. **费用**：真实 Seedance 调用按秒/分辨率计费；优先短时长、小分辨率（如 5s / 720p 或 mini 档）

### 3.7 临界余量：剩 `$0.02` 必须挡 create（回归）

复现用户实测漏洞：

```bash
# 先把 team 调到：spend≈32.68, max_budget=32.7（或任意 remaining≈0.02 << RESERVE）
curl -s -w "\nHTTP:%{http_code}\n" -X POST "http://127.0.0.1:4000/v1/videos" \
  -H "Authorization: Bearer $VIDEO_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$VIDEO_MODEL\",
    \"prompt\": \"tiny remaining must reject\",
    \"seconds\": 5,
    \"resolution\": \"720p\"
  }"
```

**验收**：

- [ ] HTTP **429** BudgetExceeded（不得 200）
- [ ] team spend 不变
- [ ] 已有视频的 `GET /v1/videos/{id}` 仍可查询（不因预占失败）

---

## 5. 自动化回归（不替代 E2E）

本地改完先跑：

```bash
cd /Users/joycloud/project/litellm
.venv/bin/python -m pytest \
  tests/test_litellm/proxy/test_budget_reservation.py::test_should_reserve_ark_video_generation_cost \
  tests/test_litellm/proxy/spend_tracking/test_volcengine_video_billing.py::test_estimate_ark_video_reservation_cost_usd_720p_text_input \
  tests/test_litellm/proxy/spend_tracking/test_volcengine_video_billing.py::test_create_returns_reserved_spend_and_stores_provisional \
  tests/test_litellm/proxy/spend_tracking/test_volcengine_video_billing.py::test_finalize_settles_delta_from_provisional_reservation \
  tests/test_litellm/proxy/spend_tracking/test_volcengine_video_billing.py::test_no_charge_refunds_provisional_reservation \
  -q
```

建议套件：

```bash
.venv/bin/python -m pytest \
  tests/test_litellm/proxy/spend_tracking/test_volcengine_video_billing.py \
  tests/test_litellm/proxy/test_budget_reservation.py \
  -q
```

---

## 6. 验收签字表

| # | 用例 | 结果 | 备注（HTTP / video_id / spend 前后） |
|---|---|---|---|
| 3.1 | 单请求挂账 + finalize 补差 | ☐ Pass / ☐ Fail | |
| 3.2 | 余额不足单条 429 | ☐ Pass / ☐ Fail | |
| 3.3 | 并发只放行 1 条 | ☐ Pass / ☐ Fail | |
| 3.4 | 失败退预占 | ☐ Pass / ☐ Fail | |
| 3.5 | 参考视频估费抬高 | ☐ Pass / ☐ Fail / ☐ Skip | |
| 3.6 | 跨模态占坑 | ☐ Pass / ☐ Fail / ☐ Skip | |
| 3.7 | 剩 $0.02 硬拒 create | ☐ Pass / ☐ Fail | |

测完恢复：

```bash
curl -s -X POST "http://127.0.0.1:4000/team/update" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"team_id":"<TEAM_ID>","max_budget": <ORIGINAL_MAX>}'
```
