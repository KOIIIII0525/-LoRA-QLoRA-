# 结果样例摘录

本文档用于记录少量可见结果摘录。完整本地预测文件位于 `results/`，但该目录下的 JSON/JSONL 文件默认被 `.gitignore` 忽略，避免纳入大量实验产物。

## 自动评估指标

评估文件：

```text
data/processed/test_100.jsonl
```

预测与指标文件：

```text
results/predictions_qwen05b_qlora_3k.jsonl
results/metrics_qwen05b_qlora_3k.json
```

| 指标 | 结果 |
|---|---:|
| examples | 100 |
| ROUGE-L | 0.2539 |
| eval loss | 1.8221 |
| perplexity | 6.1851 |

说明：ROUGE-L、eval loss 和 perplexity 只作为辅助指标，不能单独代表事实正确性或真实指令遵循能力。

## 固定 Prompt 对比样例

完整本地文件：

```text
results/manual_compare_qwen05b_qlora_3k.jsonl
```

### 样例 1：结构化方案生成

Prompt：

```text
分析公司内部沟通问题，提出改善方案
```

观察：

- Base Model 开头出现身份说明，随后给出较泛化的管理建议。
- QLoRA-3k Adapter 直接输出编号方案，结构更清楚，但受 `max_new_tokens=128` 影响存在截断。

结论：Adapter 在结构化输出上更明显，但回答完整性仍受生成长度限制。

### 样例 2：日期抽取

Prompt：

```text
从给定的文本中提取所有日期，并生成一个日期列表。
2022年春季本学期从2022年3月1日开始，4月30日结束。
```

观察：

- Base Model 给出时间范围，但夹杂额外说明。
- QLoRA-3k Adapter 明确列出 `3月1日` 和 `4月30日`。

结论：Adapter 在部分抽取类任务上更贴近指令形式。

### 样例 3：失败案例

Prompt：

```text
对于天文学中的太阳系，用简洁扼要的语言进行全面概括
```

观察：

- Base Model 存在明显事实错误。
- QLoRA-3k Adapter 同样存在事实错误，例如把太阳描述成气体行星或最大行星。

结论：小规模通用中文 SFT 没有解决事实可靠性问题，不能把 loss 下降解释为知识能力显著提升。

## 第一版保守结论

在 RTX 3060 Laptop 6GB 约束下，项目已完成 Qwen2.5-0.5B 的 QLoRA 中文指令微调和评估闭环。小规模 SFT 后，模型在部分固定 prompt 上表现出更强的结构化输出和任务导向回答倾向；但仍存在事实错误、概念混淆和生成截断，因此当前结果应表述为“观察到部分改善”，而不是“能力显著提升”。
