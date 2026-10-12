# 回答质量优化：检索范围前置（2026-09-27）

## 调研与取舍

- [WeKnora 检索引擎](https://github.com/Tencent/WeKnora/blob/main/website-docs/03-features/05-retrieval-engines.md)：检索范围、独立向量/关键词召回、上层 RRF。参考这些分层思路，不复制代码或迁移框架。
- [Haystack Rankers](https://docs.haystack.deepset.ai/docs/rankers)：检索后可按相关性重新排序。reranker 只能处理已经召回的候选，不能补回之前被截断的资料，因此本次先修漏检。
- 不增加控制台功能，不改写原始知识，不引入新的付费模型或依赖。

## 发现的确定性缺陷

原 GeneralManualKnowledge 把 top_k 提到 20，但 SqliteHybridKnowledge 使用
`min(query.top_k, configured_final_k)`，实际最多返回默认 5 条。
在这 5 条中再过滤通用手册，会让排在后面的合格证据消失。
单纯提高外层 top_k 无法解决，甚至全局 20 条候选中也可能没有合格证据。

修复：请求携带允许文档 ID，在向量/BM25 候选排名前过滤，再按原候选预算和
Top K 排序。范围只取交集、空集合拒绝所有文档，保留外层防御检查。
范围字段通过 LangChain 和跨语言查询的 dataclass replace 保留。
不放宽通用手册审核范围，不声明这些资料适用于某个具体型号。
当前实现是读取 SQLite 行后、计算排名前过滤，不宣称 SQL 索引加速。

## 验证

- 对抗回归：25 份高排名未审核资料挤占候选时，修复前漏掉唯一允许文档，修复后命中。
  这些数据只在 pytest 临时目录中创建，不进入真实知识库。
- 验证范围交集、空范围、中文翻译传递和请求间隔离。
- Python 全量：667 passed，10 skipped。
- 真实 eManual 索引版本：`gen-ad8aa67ac1b70f78`。
- 六个公开问题：灰度、图像自检、声音自检、型号/软件版本、花屏、无声。
  使用真实 Embedding，修复前后预期文档命中均为 **6/6**。
  这是小规模检索冒烟集，不是回答正确率，更不是统计上的质量提升证明。
- 修复后候选通常更多，也可能引入额外噪声。没有因为此次结果直接上线 reranker；
  后续应对最终回答的相关性、证据支持和拒答行为进行独立对比。

复现真实检索对比（会产生 6 条查询的 Embedding 费用，无回答生成、无知识写入）：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
py -3.12 scripts/evaluate_manual_scope.py --live
```

本次代码修改未重启已有手机后端，不打断用户会话；旧进程需要重启才能加载修复。
检索正确性修复已完成，但最终客服效果仍需真实多轮回答评测，不应写成简历中的准确率提升。
