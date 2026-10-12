# 共用客服 API

基地址示例 `http://127.0.0.1:8768`。下面的值仅为示例；实际口令通过 `RAG_DEMO_INVITE` 配置。

## 建立连接

`POST /demo/session`，`Content-Type: application/json`：

```json
{"invite": "YOUR_PRIVATE_INVITE_AT_LEAST_16_CHARS"}
```

成功返回 `token` 与当前 `mode`、`model`、`embedding`、`dataset`、`document_count`、`source`、`locales`。不会返回模型 API Key。后续请求添加 `Authorization: Bearer <token>`。

## 提交和接收消息

`POST /demo/messages`：

```json
{"request_id": "message-00000001", "text": "三星电视怎么打开图像测试？", "locale": "zh-CN"}
```

返回 HTTP 202 和 `{"job_id":"message-00000001"}`。`request_id` 允许 8–80 个英文字母、数字、下划线或连字符。相同编号和相同内容不会重复调用模型；相同编号不同内容返回 409。

随后 `GET /demo/messages/message-00000001`：

- `{"status":"pending"}`：尚在处理，继续轮询。
- `{"status":"done","result":{...}}`：处理完成，读取完整回复。
- `{"status":"failed","error":"..."}`：生成失败或超时，前端提示错误。

`done` 不等于成功回答；还需检查 `result.status`。结果包含 `reply`、`status`、`reason`、`sources`、`evidence_count`、`stage`。`status` 可能为 `replied`、`handed_off` 或 `human_active`；后两者可能没有 `reply`，由前端呈现适当的人工接管状态，当前仅为模拟人工。

推荐结果可能附有已检索的网页来源，知识手册引用主要用于内部审计。`evidence_count` 是引用数量，不是回答正确率。

## 对话控制

`POST /demo/resume` 和 `POST /demo/new-conversation` 都需 Bearer Token，无需消息内容。有任务尚在运行时返回 409。恢复 AI 保留原对话；新建对话清空上下文。

## 错误与部署范围

401 表示缺失、错误或过期的会话 Token，重新建立连接；403 表示口令错误；409 表示当前忙碌或请求编号冲突；422 表示参数不正确；429 表示演示限流或额度达到上限。

会话闲置一小时过期，重启也会清空会话。每个会话同时处理一条消息。异步生成最多等待 180 秒；原有 HTTP 演示行为保持不变。

网页同步接口通过同源 Cookie 使用 `POST /api/session`、`POST /api/chat`、`POST /api/resume`，请求头需 `X-Chat-Client: 1`。`/api/chat` 直接接收 `text` 和 `locale`，无 `request_id`，返回完整结果。它限制 `Host` 为 `localhost` 或 `127.0.0.1`；外部前端优先使用自己的服务端代理调用 Bearer 接口。

`--public-demo` 不暴露 `/api/*`，避免绕过演示口令和额度；无论哪种模式均不提供 HTML、JavaScript、CSS、WXML 或 WXSS。
