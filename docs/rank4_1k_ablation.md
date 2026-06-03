# Rank4-1k 消融实验记录

本文档记录 `rank=4`、`train_1k` 的 LoRA/QLoRA 消融结果。该实验用于和已完成的 `rank=8`、`train_1k` 训练结果做低风险对比，并为后续 `rank=4`、`train_3k` 主消融实验做准备。

## 实验配置

| 配置项 | rank4-1k | rank8-1k 对照 |
|---|---|---|
| base model | `models/qwen2.5-0.5b-instruct` | `models/qwen2.5-0.5b-instruct` |
| train split | `data/processed/train_1k.jsonl` | `data/processed/train_1k.jsonl` |
| valid split | `data/processed/valid_300.jsonl` | `data/processed/valid_300.jsonl` |
| train rows | 1000 | 1000 |
| epoch | 1 | 1 |
| max seq len | 512 | 512 |
| batch size | 1 | 1 |
| gradient accumulation | 8 | 8 |
| learning rate | 2e-4 | 2e-4 |
| LoRA rank | 4 | 8 |
| LoRA alpha | 8 | 16 |
| output dir | `outputs/qwen05b_qlora_1k_r4/` | `outputs/qwen05b_qlora_1k/` |

rank4-1k 启动命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b_rank4.yaml --train_split train_1k --output_dir outputs\qwen05b_qlora_1k_r4
```

## 训练结果

| 实验 | step | logged train loss 均值 | 最后一次 logged train loss | valid eval loss | 产物 |
|---|---:|---:|---:|---:|---|
| QLoRA-1k-rank4 | 125 / 125 | 约 1.8500 | 约 1.7987 | 约 1.7739 | 已生成 adapter |
| QLoRA-1k-rank8 | 125 / 125 | 约 1.8355 | 约 1.7923 | 约 1.7662 | 已生成 adapter |

训练阶段观察：

- rank4-1k 完成完整 1 epoch，没有中途停止。
- rank4-1k 的最终 valid eval loss 约 `1.7739`，略高于 rank8-1k 的约 `1.7662`。
- 在相同数据、epoch、batch、学习率和验证集下，rank8 在训练指标上略优，但差距较小，应保守表述为“小幅差异”。

## 自动评估

rank4-1k 已按第一版主线相同口径在 `test_100` 上完成自动评估。

运行命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval_rank4_1k.yaml --skip_perplexity --max_new_tokens 128 --predictions_file results\predictions_qwen05b_qlora_1k_r4.jsonl --metrics_file results\metrics_qwen05b_qlora_1k_r4_rouge_only.json

D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval_rank4_1k.yaml --skip_generation --predictions_file results\predictions_qwen05b_qlora_1k_r4.jsonl --metrics_file results\metrics_qwen05b_qlora_1k_r4.json
```

输出文件：

| 文件 | 行数 / 状态 | 说明 |
|---|---:|---|
| `results/predictions_qwen05b_qlora_1k_r4.jsonl` | 100 行 | test set 预测 |
| `results/metrics_qwen05b_qlora_1k_r4_rouge_only.json` | 已生成 | ROUGE-L 快照 |
| `results/metrics_qwen05b_qlora_1k_r4.json` | 已生成 | ROUGE-L + eval loss + perplexity |

自动指标：

| 实验 | test examples | ROUGE-L | test eval loss | perplexity |
|---|---:|---:|---:|---:|
| QLoRA-1k-rank4 | 100 | 0.2195 | 1.8499 | 6.3592 |
| QLoRA-3k-rank8 主线 | 100 | 0.2539 | 1.8221 | 6.1851 |

说明：当前仓库已有完整自动指标的是 `rank8-3k` 主线结果，而不是 `rank8-1k`。因此上表只能说明 rank4-1k 与主线 3k-rank8 的差距，不能作为严格的 rank4-vs-rank8 同数据量自动指标结论。严格的 rank4-vs-rank8 自动指标对比，需要后续补跑 `outputs/qwen05b_qlora_1k/` 的 `test_100` 评估。

## 固定 Prompt 对比

rank4-1k 已在 `manual_prompts_30.jsonl` 上生成 Base vs Adapter 对比。

运行命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval_rank4_1k.yaml --input_file data\processed\manual_prompts_30.jsonl --output_file results\manual_compare_qwen05b_qlora_1k_r4.jsonl --mode both --max_new_tokens 128
```

完整性检查：

| 检查项 | 结果 |
|---|---:|
| 样本数 | 30 |
| base 输出缺失数 | 0 |
| adapter 输出缺失数 | 0 |
| base 平均输出长度 | 约 195.1 字符 |
| adapter 平均输出长度 | 约 197.4 字符 |

初步样例观察：

| 样例 | prompt 摘要 | rank4-1k adapter 观察 |
|---|---|---|
| `manual_8244` | 公司内部沟通方案 | 给出编号式改善建议，结构化倾向明显，但仍可能被长度上限截断 |
| `manual_1586` | 介绍川菜 | 能生成连续中文段落，覆盖技艺和食材，但内容较泛 |
| `manual_2284` | 描述公园景色 | 输出流畅，偏模板化描写，事实风险较低但个性化不足 |

## 当前结论

1. rank4-1k 已完成训练、自动评估和固定 prompt 对比，输出文件命名独立，没有覆盖 rank8 主线结果。
2. 在 `train_1k / valid_300` 的训练指标上，rank4 的 valid eval loss 约 `1.7739`，rank8 约 `1.7662`，rank8 略优。
3. rank4-1k 的 `test_100` 自动指标为 ROUGE-L `0.2195`、test eval loss `1.8499`、perplexity `6.3592`。
4. 当前不能把 rank4-1k 自动指标直接和 rank8-1k 比，因为 rank8-1k 的 `test_100` 自动评估尚未补跑。若要做严格 rank 消融表，下一步应补跑 rank8-1k 同口径评估，或继续完成 rank4-3k 后与 rank8-3k 主线做同数据规模对比。

## 下一步建议

优先继续 `rank=4`、`train_3k` 主消融实验。原因是仓库已有 rank8-3k 的完整训练、自动评估和固定 prompt 结果；补齐 rank4-3k 后，可以直接做同数据量、同评估集、同输出格式的主消融对比。
