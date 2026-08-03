# Rezecyan 模型服务商接入方案

> 状态：已确认并实施（代码由人工提交，AI 不 commit / push）
> 日期：2026-07-28
> 涉及仓库：litellm（fork）、erp、base；joycloud-web 仅列前端修改点（无代码）

## 一、背景与目标

接入新的模型服务提供商 Rezecyan：

- Base URL：`https://api.rezecyan.com/v1`（标准 OpenAI 兼容端点）
- 鉴权：API Key（Bearer）
- 请求/响应：原生 OpenAI 格式；chat 响应额外携带
  `message.reasoning_content`、`usage.completion_tokens_details.reasoning_tokens`

目标：

1. LiteLLM 侧以一等公民 `rezecyan/<model>` 前缀接入，**含完整定价**（走本地价格表）
2. 参照 DashScope，同步完成 erp（模型目录/账号/部署/折扣）与 base（计费 hook）配套改动
3. 本期端点：chat / embeddings / rerank / images/generations

## 二、已确认结论（原第六节）

| # | 问题 | 结论 |
|---|------|------|
| 1 | 模型清单与官方价格 | `scripts/sync_rezecyan_prices.py --scrape` 用 Playwright 从定价页拉取（只取 `ali-of-pro` 分组，**需先 `--login` 保存登录态**——该分组倍率与弹窗面板仅登录后下发，未登录只能看到 default 分组价，脚本此时直接报错不回退），写入 `rezecyan_prices.json` 并同步两份 `model_prices` JSON（先清旧 `rezecyan/*` 再写入去重） |
| 2 | reasoning tokens 计费 | **不另计费**；已含在 `completion_tokens` 内，价格表不写 `output_cost_per_reasoning_token` |
| 3 | 缓存命中价 | **有**：缓存读取 / 缓存写入分别入表（`cache_read_input_token_cost` / `cache_creation_input_token_cost`） |
| 4 | 端点范围 | chat + embeddings + rerank + images/generations |
| 5 | 多轮 `reasoning_content` 回传 | **需要**（Rezecyan 上也有 DeepSeek 模型）；chat 侧对模型名含 `deepseek` 的请求做 `_fill_reasoning_content` |
| 6 | 多地域/多档位定价 | **无**；单一 endpoint、单一价格档 |
| 7 | provider 命名 | LiteLLM 前缀 `rezecyan`，erp 常量 `"Rezecyan"` |
| 8 | 前端 | 本期**只列修改点**供前端开发，不改 joycloud-web 代码 |

币种：统一 **USD 入价格表**；爬取时默认采用站点 `usd_exchange_rate`（当前为 **7.0**），可在源表改 `cny_per_usd` 后直接再跑一遍脚本（无 `--scrape`）重生成。

## 三、改动落地

### 3.1 litellm fork

新增：

```
litellm/llms/rezecyan/
  __init__.py
  common_utils.py
  chat/transformation.py          # OpenAI 兼容 + DeepSeek reasoning 回传
  embed/transformation.py
  rerank/transformation.py        # 路径 /v1/rerank
  image_generation/transformation.py  # 路径 /v1/images/generations
scripts/rezecyan_prices.json
scripts/sync_rezecyan_prices.py
tests/test_litellm/llms/rezecyan/...
```

注册点（照 DashScope）：`LlmProviders.REZECYAN`、`constants.py` openai_compatible_*、
`get_llm_provider_logic.py`、`utils.py` ProviderConfigManager（chat/embed/rerank/image）、
`__init__.py` 模型集合 + 懒加载、`Dockerfile.volcengine` overlay。

定价示例（`qwen3.7-plus`，`ali-of-pro`：≤256k 为 ¥2/¥8/缓存读 ¥0.2/缓存写 ¥2.5；>256k 写入 `*_above_256k_tokens`）：

```bash
pip install playwright && playwright install chromium
python3 scripts/sync_rezecyan_prices.py --self-test
python3 scripts/sync_rezecyan_prices.py --login    # 首次：浏览器登录并保存登录态
python3 scripts/sync_rezecyan_prices.py --scrape   # 爬取 + 写入两份价格 JSON
```

### 3.2 erp

新增 `api/service/litellm/rezecyan.go` + `rezecyan_test.go`：

- `buildRezecyanModelRow`：`base_model` 必须 `rezecyan/` 前缀
- `buildRezecyanDeploymentRequest`：`litellm_params.model` + `api_key` + `api_base`（空则默认官方地址）
- `model_info.base_model` = 价格表 key；`mode` = Text→chat / Image→image_generation

散点：

| 文件 | 改动 |
|------|------|
| `bedrock.go` | `litellmProviderRezecyan = "Rezecyan"` |
| `video.go` | `normalizeLitellmProvider` 增加 `rezecyan` |
| `dashscope.go` | `isAPIKeyAccountProvider` / `ensureAPIKeyModel` / `apiKeyProviderFromCatalog` / `syncAPIKeyDeploymentToLiteLLM` 分派 Rezecyan |
| `account_setup.go` | apiKey 必填；apiBase 可空并默认官方地址 |
| `bedrock_catalog.go` | Create / Update / Delete / modelType 分支 |
| `discount_rule.go` | **无需改**：无档位，走通用 `custom[baseModel]=rate` |
| `docs/api/openapi-*.json` | provider 枚举补 `DashScope`（补齐存量）+ `Rezecyan` |

数据库：无 SQL 迁移。上线时管理端插入 `tb_litellm_provider_region`（Rezecyan + 逻辑 region）、账号与模型目录。

### 3.3 base

| 文件 | 改动 |
|------|------|
| `litellm_ext/custom_billing.py` | **零改动**；`rezecyan/` key 经 `model_info.base_model` 精确匹配即可 |
| `litellm_ext/test_custom_billing_discount.py` | 补 Rezecyan 折扣回归（场景 11–13） |
| `eksconf/litellm.yaml` | 构建新 fork 镜像后更新 tag（人工） |

### 3.4 joycloud-web erp-admin 前端修改指南（本期由前端开发，本方案不改代码）

仓库：`joycloud-web`，应用：`apps/erp-admin`。  
**对照物**：DashScope 已有完整链路；Rezecyan 与它同属「API Key 账号」供应商，但有 3 处关键差异，不要原样复制。

#### 3.4.1 命名与后端契约（先对齐再写 UI）

| 场景 | 前端展示枚举 | 提交给 erp 的值 | 说明 |
|------|-------------|----------------|------|
| 供应商下拉 / 列表展示 | `Rezecyan` | 账号创建：`"Rezecyan"`（见 `toCreateAccountProvider`） | erp 常量 `litellmProviderRezecyan = "Rezecyan"` |
| 区域查询、模型列表筛选等 | — | `"rezecyan"`（小写） | 走 `toLlmProviderApiValue`，与 DashScope→`dashscope` 同模式 |
| LiteLLM 价格表 / 模型 `baseModel` | 表单填写 | `rezecyan/<model>`，如 `rezecyan/qwen3.7-plus` | **必须**带此前缀，erp `buildRezecyanModelRow` 会校验 |
| 账号鉴权 | — | `authType: "api_key"` | 与 Volcano / BytePlus / DashScope 相同；**不要**走 `assume_role` |
| 账号 API Key | 必填 | `apiKey` | |
| 账号 API Base | **选填** | `apiBase` 可空；空则后端默认 `https://api.rezecyan.com/v1` | **与 DashScope 必填不同** |
| 模型类型 | Text / Image / embedding / rerank | 同左 | 无 Video；Text 在 litellm 侧映射为 `chat` |
| 价格档 / 地域后缀 | 无 | 不要在 `baseModel` 后加 `-cn/-hk/-eu` | 单一价格档 |

#### 3.4.2 与 DashScope 的差异（避免抄错）

| 点 | DashScope | Rezecyan |
|----|-----------|----------|
| API Base | api_key 模式下**必填**（多地域 / WorkspaceId 域名） | **选填**；placeholder 给官方默认即可 |
| `baseModel` 提示 | 要说明价格档后缀（`-cn` 等） | 只要求 `rezecyan/<model>`，**不要**引导填档位后缀 |
| 区域管理下拉 | `provider-options.ts` 里单独把 DashScope 补进区域厂商 | 同样把 Rezecyan 补进（`CLOUD_VENDORS` 里没有它） |
| 折扣 key | 按地域派生多档 key | 前端无感；折扣 key 即 `baseModel` |

#### 3.4.3 按文件改什么

**1. 常量** — `apps/erp-admin/utils/constants.ts`

```ts
// aws-bedrock, byteplus, dashscope, rezecyan
export const LLM_PROVIDERS = ['Bedrock', 'BytePlus', 'DashScope', 'Rezecyan'] as const
```

`LlmProvider` 由 `as const` 推导，会自动带上 `Rezecyan`。若 `LlmModelOwner` 需要展示 Rezecyan 上的厂商（如 DeepSeek），按产品需要再扩，非接入硬性要求。

**2. Provider 双向映射** — `apps/erp-admin/infrastructure/llm-model/provider-mapper.ts`

在现有 DashScope 旁增加：

```ts
const LLM_PROVIDER_API_VALUE: Record<LlmProvider, string> = {
  // ...
  DashScope: 'dashscope',
  Rezecyan: 'rezecyan',
}

const LLM_PROVIDER_UI_BY_API: Record<string, LlmProvider> = {
  // ...
  dashscope: 'DashScope',
  rezecyan: 'Rezecyan',
}

/** 是否为 Rezecyan（api_key 下 API Base 选填，与 DashScope 必填区分） */
export function isRezecyanLlmProvider(provider?: string): boolean {
  return toLlmProviderApiValue(provider ?? '') === 'rezecyan'
}
```

`toLlmProviderApiValue` / `toLlmProviderUiValue` 读上述 map，一般不用再改逻辑。

**3. 区域管理选项** — `apps/erp-admin/views/llm-management/llm-exclusive-model/provider-options.ts`

`buildRegionProviderOptions` 当前只把不在 `CLOUD_VENDORS` 里的 `DashScope` 拼进区域厂商下拉。改为同时补 `Rezecyan`，例如：

```ts
const EXTRA_REGION_LLM_PROVIDERS = ['DashScope', 'Rezecyan'] as const
const llmProviderOptions = LLM_PROVIDERS.filter(provider =>
  EXTRA_REGION_LLM_PROVIDERS.includes(provider as (typeof EXTRA_REGION_LLM_PROVIDERS)[number]) &&
  !CLOUD_VENDORS.some(vendor => vendor.toLowerCase() === provider.toLowerCase())
).map(/* 同现有 map */)
```

否则运营在「区域管理」里建 `tb_litellm_provider_region` 时选不到 Rezecyan。

**4. 账号创建/编辑弹窗** — `.../add-llm-exclusive-account-dialog.tsx`

- 供应商下拉已吃 `LLM_PROVIDERS`，加常量后会出现 Rezecyan。
- API Base 校验：今日逻辑是「仅 DashScope + api_key → 必填」。**不要**把 Rezecyan 并进必填；保持选填。
- 建议：Rezecyan + api_key 时显示弱提示（非红色 *），placeholder / hint 指向官方地址。

伪代码：

```ts
const isDashScopeApiKey = showApiKeyFields && isDashScopeLlmProvider(cloudProvider)
const isRezecyanApiKey = showApiKeyFields && isRezecyanLlmProvider(cloudProvider)

// 校验：仅 DashScope 必填
rules={{
  validate: value =>
    !isDashScopeApiKey || value.trim() || t('validation.apiBaseRequired'),
}}

// UI：Rezecyan 给可选提示
{isRezecyanApiKey && <p className="text-xs">{t('hint.rezecyanApiBase')}</p>}
```

**5. 账号提交时的 provider 映射** — `.../llm-exclusive-model/index.tsx` 的 `toCreateAccountProvider`

现有：

```ts
if (cloudProvider === 'DashScope') return 'DashScope'
```

补一行：

```ts
if (cloudProvider === 'Rezecyan') return 'Rezecyan'
```

漏了的话，可能把 UI 枚举原样或小写错误提交，erp 虽能 normalize，但列表回显/筛选会别扭。

**6. 模型目录弹窗** — `.../llm-list/components/add-llm-model-dialog.tsx`

- 供应商选项来自 `LLM_PROVIDERS`，常量加上即可。
- `baseModel` 前缀：现有 `resolveBedrockModelPrefix` 对非 Bedrock 已是 `` `${toLlmProviderApiValue(provider)}/` ``，选 Rezecyan 时会得到 `rezecyan/`，一般无需特判。
- 模型类型：现有选项已含 Text / Image / embedding / rerank；**不要**为 Rezecyan 放开 Video。若产品要限制可选类型，可在 `vendor === 'Rezecyan'` 时过滤掉 Video / Audio。
- 价格档 hint：今日 `hint.baseModelPricingTier` 文案以 DashScope 档位为例。建议按供应商切换文案，或增加 Rezecyan 专用 hint：

  - DashScope：继续说明 `-cn/-hk/-eu`
  - Rezecyan：`请填写价格表 key，例如 rezecyan/qwen3.7-plus（无地域档位后缀）`

**7. i18n（中英文都要补）**

至少改这些文件里的 `provider.*`（以及账号 hint）：

| 文件 | 键 | 建议文案 |
|------|-----|----------|
| `messages/zh/llmList.json`、`en/llmList.json` | `provider.rezecyan` | `Rezecyan` |
| 同上 | `hint.baseModelRezecyan`（新建）或扩展现有 hint | 见上「无档位后缀」说明 |
| `messages/zh/llmExclusiveModel.json`、`en/llmExclusiveModel.json` | `provider.rezecyan` | `Rezecyan` |
| 同上 | `hint.rezecyanApiBase`（新建） | 中：`可选。留空则使用官方地址 https://api.rezecyan.com/v1`；英：`Optional. Leave empty to use https://api.rezecyan.com/v1` |
| `messages/zh/regionList.json`、`en/regionList.json` | `provider.rezecyan` | `Rezecyan` |
| `messages/zh/modelAccountProfit.json`、`en/modelAccountProfit.json` | `provider.rezecyan` | `Rezecyan`（若账单/利润筛选项吃同一套 provider） |
| `messages/zh/common.json`、`en/common.json` | 若有 `cloudVendor` 大写枚举习惯 | 视区域管理是否复用；优先保证 `regionList.provider.rezecyan` |

原则：凡是今天已有 `provider.dashscope` 的消息文件，同步加 `provider.rezecyan`，避免下拉显示 key 原文。

#### 3.4.4 前端自测清单

1. **区域管理**：可创建 provider=`rezecyan`（或展示名 Rezecyan）的区域；列表能正确显示名称。
2. **账号**：选 Rezecyan + api_key → API Key 必填、API Base 可不填仍能提交；填自定义 Base 也能提交。
3. **模型目录**：选 Rezecyan，`baseModel` 自动/手动为 `rezecyan/...`；类型选 Text/Image/embedding/rerank 可保存；Video 不应作为推荐路径。
4. **筛选/回显**：账号列表、模型列表、区域列表里选中或展示 Rezecyan 不落到 `—` 或原始小写串。
5. **回归**：DashScope 的 API Base 必填、价格档 hint **行为不变**。

#### 3.4.5 不阻塞后端

前端未合入前，运营可用 erp OpenAPI / 管理接口直接建 region、账号、模型目录；后端已支持 `provider=Rezecyan`。

## 四、汇率维护

1. **单一汇率源**：`scripts/rezecyan_prices.json` 的 `cny_per_usd`（爬取默认跟站点；可手工改）
2. **生成脚本**：`scripts/sync_rezecyan_prices.py`（`--scrape` 拉取 `ali-of-pro`）换算写入价格 JSON，条目附审计字段（原 CNY、汇率、as_of）
3. **调整节奏**：不追每日即期价；季度审查或偏离 ±3% 时改源文件一个数字 → 重跑脚本 → 重建镜像
4. **历史账不追溯**：spend log 是计费时刻快照；对账用源文件人民币原价

## 五、验收清单

1. litellm：`get_llm_provider("rezecyan/qwen3.7-plus")` → provider=`rezecyan`；单测 `tests/test_litellm/llms/rezecyan/` 通过；`completion_cost` 与牌价×汇率一致
2. 镜像：`Dockerfile.volcengine` overlay 含 `llms/rezecyan/`；dev 部署后 curl chat，spend log cost ≠ 0
3. erp：`go test ./...` 通过；管理端/API 走通「建 region → 建账号 → 建模型 → 部署」
4. base：`python3 litellm_ext/test_custom_billing_discount.py` 通过；team `custom_discount` 对 `rezecyan/<model>` 生效

## 六、工作量（实际）

| 仓库 | 内容 | 估时 |
|------|------|------|
| litellm fork | provider（chat/embed/rerank/image）+ 注册 + 定价脚本 + Dockerfile + 单测 | ~1 天 |
| erp | rezecyan.go + 散点分支 + 表驱动测试 + OpenAPI | ~1–1.5 天 |
| base | 折扣回归（近零业务代码） | ≤0.5 天 |
| erp-admin 前端 | 按 3.4 对照 DashScope 补齐（常量/映射/账号 Base 选填/i18n/区域） | ~0.5 天 |
| 联调 + dev 验证 | 端到端计费/折扣 | 0.5–1 天 |
