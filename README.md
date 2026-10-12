# RAG 智能客服后端

网页聊天和微信小程序共用的 Python 后端，包含对话理解、上下文记忆、混合检索、模型回复校验和 HTTP API。此仓库不包含网页页面、微信小程序界面或知识检查台。

从 `Viki0258/whatsapp-customer-service-rag` 的当前工作区提取，包含已完成但尚未提交的后端改进。原项目和已有前端保持独立；Python 包名 `whatsapp_support` 及 `/api/*`、`/demo/*` 路径保留兼容性。

## 已有能力

- 使用 LangChain Core 自定义 Retriever 和 LCEL 组织检索、Prompt、生成和校验；Qwen Embedding 与 DeepSeek 回复模型分别配置。
- SQLite 知识索引，向量检索 + BM25 + RRF，知识 generation 原子切换与未变化内容的 Embedding 缓存。
- 中英文提问、跨语言检索；对话理解、检索和回答共享上下文，保留最近六轮原文及较早轮次的有界摘要。
- 型号和故障澄清、基于已审核通用手册的排查、三星产品联网推荐；联网能力取决于实际配置的服务和页面可访问性。
- 完整生成并校验后返回回复，包含内部引用校验、来源记录、危险操作拦截和模拟人工接管。
- Cookie 同步接口与 Bearer 异步任务接口调用同一服务；异步接口支持任务轮询和请求编号去重。

当前 HTTP 演示会话、消息任务和人工状态保存在内存中，使用单进程；重启后需重新连接。PostgreSQL、Redis 与 WhatsApp 适配器是保留的后端模块，并不代表这个演示 HTTP 服务已经采用持久化会话。真实订单、退款和人工客服接入需要业务系统支持。

## 本地运行

Python 3.12，在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install -e ".[test,runtime,evaluation]"
$env:RAG_DEMO_INVITE = '<至少16字符的自定义连接口令>'
rag-backend --dataset sample --port 8768
```

默认 fake-first，不调用模型或下载数据。`knowledge/` 是用于验证流程的合成资料，并非真实商店政策。

启用真实模型时，把 `examples/` 的两个配置模板分别复制为 `.rag-data/bailian.local.json` 和 `.rag-data/deepseek.local.json`，在本机填入 API Key、已开通的模型名及对应向量维度。运行：

```powershell
rag-backend --live --dataset sample --port 8768
```

可通过 `--embedding-config`、`--response-config`、`--knowledge-dir` 指定配置和自有知识目录。配置文件、完整数据集、索引和日志均被 Git 忽略，仓库仅提供无真实密钥的模板。

## 使用公开手册数据

完整数据不随代码上传。重新准备此前使用的语料：

```powershell
py -3.12 scripts/datasets/download_public.py multidoc2dial .rag-data/public/multidoc2dial.zip --allow-network
py -3.12 scripts/datasets/download_public.py ragbench-emanual .rag-data/public/emanual-test.parquet --allow-network
py -3.12 scripts/datasets/download_public.py bitext-customer-support .rag-data/public/bitext.csv --allow-network
py -3.12 -m scripts.datasets.prepare_public --output .rag-data/prepared-public-v3
rag-backend --live --dataset emanual --port 8768
```

准备工具检查下载来源、哈希、去重与评测标注。已有生成目录不能覆盖；需要重新准备时另选目录并用 `--prepared-root` 启动。许可证及来源见 [公开数据说明](docs/evaluation/datasets.md)。eManual 是通用手册片段，未建立可靠的具体电视型号对应关系；不能把任意型号视为已获得专属说明书。

## 前端接入

网页与小程序均可使用以下接口：

| 接口 | 用途 |
| --- | --- |
| `GET /healthz` | 进程健康检查，成功返回 `{"status":"ok"}`，不验证模型服务在线状态 |
| `POST /demo/session` | 提交连接口令，获得会话 Token |
| `POST /demo/messages` | 携带 Token 提交消息，获得任务编号 |
| `GET /demo/messages/{job_id}` | 轮询完整回答 |
| `POST /demo/resume` | 恢复 AI，保留当前上下文 |
| `POST /demo/new-conversation` | 新建对话，清空该会话上下文 |

仍保留 `/api/session`、`/api/chat`、`/api/resume`，适用于同源网页或反向代理；`POST` 需 `X-Chat-Client: 1` 且保持 Cookie。独立网页需由自己的同源后端代理调用，当前服务未开放跨域浏览器请求。

通过 HTTPS 反向代理或隧道用于手机演示时，使用 `--public-demo`，仅开放口令保护的 `/demo/*` 和 `/healthz`，带登录频率、并发和消息额度限制。默认监听 `127.0.0.1`。该仓库不包含临时穿透地址或真实演示口令。

请求示例、响应和错误码见 [API 接入说明](docs/api.md)。

## 验证与历史适配器

```powershell
py -3.12 -m pytest -q
rag-backend --help
git diff --check
```

默认测试使用假模型和本地临时存储，不调用真实 API。`scripts/evaluate_customer_dialogue.py --live` 和 `scripts/evaluate_manual_scope.py --live` 是需主动运行的付费联调脚本，不进入默认 CI。过去的联调记录位于 `docs/evaluation/`，其日期和结果不代表新仓库已重新执行相同联调。

PostgreSQL, Redis, OpenAI embedding, and OpenAI response adapters exist. In the default offline suite, live external integration is not exercised or claimed. Optional durable worker setup and integration checks are documented in [local development](docs/local-development.md); RAG configuration and safety contracts are in [RAG development](docs/rag-local-development.md).

