# DPO 偏好对齐阶段规划

本文档作为换对话后的项目记忆，用于记录当前项目如何从已完成的 QLoRA SFT 闭环扩展到 DPO 偏好对齐。

## 项目新定位

项目升级为：

```text
基于 Qwen2.5-0.5B 的中文指令微调与偏好对齐实验项目
```

整体流程：

```text
数据处理 -> QLoRA SFT -> Base/SFT 评估 -> 偏好数据构造 -> DPO 对齐 -> Base/SFT/DPO 对比
```

## 已有 SFT 基线

| 项目 | 当前结果 |
|---|---|
| base model | `models/qwen2.5-0.5b-instruct` |
| SFT adapter | `outputs/qwen05b_qlora_3k/` |
| SFT train split | `data/processed/train_3k.jsonl` |
| SFT valid eval loss | 约 `1.7448` |
| test ROUGE-L | 约 `0.2539` |
| test eval loss | 约 `1.8221` |
| test perplexity | 约 `6.1851` |

当前结论仍保持保守：小规模 QLoRA SFT 改善了部分结构化输出和指令遵循倾向，但没有解决事实错误、概念混淆和生成截断。

## DPO 数据策略

第一版不做人工大规模标注，也不做 PPO。采用轻量离线偏好对齐：

```text
chosen = BELLE 原始参考答案
rejected = QLoRA-3k SFT adapter 生成答案
```

偏好样本格式：

```json
{
  "id": "pref_0001",
  "prompt": "用户指令",
  "chosen": "参考答案",
  "rejected": "SFT adapter 生成答案",
  "source": "reference_vs_adapter_response"
}
```

说明：这应表述为“离线偏好对齐 / RLAIF 风格实验”，不要夸大为完整人类 RLHF。

## 当前新增文件

| 文件 | 作用 |
|---|---|
| `configs/dpo_qwen_0.5b.yaml` | DPO 阶段配置、偏好数据路径和后续训练参数 |
| `src/preference_utils.py` | 偏好样本抽取、校验和划分工具 |
| `scripts/prepare_preference_data.py` | 生成偏好 prompt 池、将预测转成 chosen/rejected 数据 |
| `tests/test_preference_utils.py` | 偏好工具测试 |
| `tests/test_prepare_preference_data.py` | 偏好数据脚本测试 |

## 当前数据隔离审核

DPO 训练集应尽量独立于已完成的 QLoRA SFT 训练和评估闭环。当前配置已排除：

- `data/processed/train_100.jsonl`
- `data/processed/train_1k.jsonl`
- `data/processed/train_3k.jsonl`
- `data/processed/valid_300.jsonl`
- `data/processed/test_100.jsonl`
- `data/processed/manual_prompts_30.jsonl`

2026-06-02 重新生成 `preference_prompts_pool.jsonl` 后，审核结果：

| 对比 split | prompt 重叠 |
|---|---:|
| `train_100` | 0 |
| `train_1k` | 0 |
| `train_3k` | 0 |
| `valid_300` | 0 |
| `test_100` | 0 |
| `manual_prompts_30` | 0 |

说明：DPO 偏好数据现在是单独 held-out 数据，不和已完成 SFT 主实验或评估集重合。

## 当前已生成的本地产物

以下文件在 `.gitignore` 范围内，不提交：

| 文件 | 状态 |
|---|---|
| `data/processed/preference_prompts_pool.jsonl` | 已重新生成 500 行，与旧 SFT/评估 split 重叠为 0 |
| `results/preference_rejected_qwen05b_qlora_3k_quick.jsonl` | 已生成 1 行 adapter rejected 冒烟结果 |
| `results/preference_rejected_chunk_000.jsonl` | 已生成 50 行 adapter rejected |
| `data/processed/preference_train_50.jsonl` | 已生成 50 行 DPO smoke 数据 |
| `outputs/qwen05b_dpo_50/` | 已完成 DPO smoke 训练并生成 adapter |
| `outputs/qwen05b_dpo_300/` | 已完成 DPO-300 主实验并生成 adapter |

质量检查：

| 检查项 | 结果 |
|---|---:|
| preference prompt pool 行数 | 500 |
| 与旧 SFT/评估 split prompt 重叠 | 0 |
| `preference_train_50` 行数 | 50 |
| `preference_train_50` empty chosen | 0 |
| `preference_train_50` empty rejected | 0 |
| `preference_train_50` chosen/rejected 完全相同 | 0 |

## 长任务处理记录

一次性生成 500 条 rejected responses 时，后台进程正常使用 GPU，但 `infer.py` 原实现会在全部生成完成后一次性写文件，中途不可观察。该长任务已停止，改为支持：

```powershell
--offset N --limit M
```

当前建议分批生成，每批 50 条，输出到独立 chunk 文件。第 1 批命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\preference_prompts_pool.jsonl --output_file results\preference_rejected_chunk_000.jsonl --mode adapter --offset 0 --limit 50 --max_new_tokens 128
```

观察：50 条约耗时 20 分钟以上，因此后续不要再用一次性 500 条黑盒运行。第一版优先使用 `preference_train_50` 做 DPO smoke test。

## DPO Smoke Test 结果

2026-06-02 已安装 `trl==1.5.1` 到 `pytorch_env`，并完成 DPO 50 条 smoke test。

训练命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_50 --output_dir outputs\qwen05b_dpo_50
```

关键结果：

| 指标 | 结果 |
|---|---:|
| train rows | 50 |
| optimizer steps | 7 |
| train runtime | 约 38 秒 |
| train loss | 约 0.6355 |
| rewards/accuracies | 约 0.84 |
| rewards/margins | 约 0.1384 |
| output dir | `outputs/qwen05b_dpo_50/` |

产物检查：

- `outputs/qwen05b_dpo_50/adapter_model.safetensors` 已生成。
- `configs/eval_dpo_50.yaml` 已新增，用于 DPO smoke adapter 推理/评估。
- 已用 `scripts/infer.py --config configs/eval_dpo_50.yaml --mode adapter` 完成单条真实推理。

环境问题记录：

- DPO 初次训练在 `fp16=true` 下触发 `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`。
- 诊断发现 trainable LoRA 参数为 `float32`，问题来自当前 TRL/Transformers/Accelerate/PyTorch 组合下 AMP GradScaler 与 bfloat16 梯度路径不兼容。
- 已将 `configs/dpo_qwen_0.5b.yaml` 中 DPO 阶段 `fp16` 改为 `false`，仍保留 4-bit QLoRA 加载；DPO smoke test 随后通过。

## DPO-300 主实验结果

2026-06-02 已完成 `preference_train_300` DPO 主实验。

训练命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_300 --output_dir outputs\qwen05b_dpo_300
```

关键结果：

| 指标 | 结果 |
|---|---:|
| train rows | 300 |
| valid rows | 100 |
| optimizer steps | 38 |
| train runtime | 约 13 分 19 秒 |
| train loss | 约 0.419 |
| eval loss | 约 0.3049 |
| eval rewards/accuracies | 约 0.93 |
| eval rewards/margins | 约 1.2685 |
| output dir | `outputs/qwen05b_dpo_300/` |

产物检查：

- `outputs/qwen05b_dpo_300/adapter_model.safetensors` 已生成。
- `outputs/qwen05b_dpo_300/checkpoint-38/` 已生成。
- `configs/eval_dpo_300.yaml` 已新增，用于 DPO-300 adapter 推理/评估。
- 已用 `scripts/infer.py --config configs/eval_dpo_300.yaml --mode adapter` 完成单条真实推理。

注意：DPO-300 训练完成不等于最终项目结论完成；当前已补齐自动指标、fixed prompt 人工分析和 preference accuracy，最终结论仍需保守表达。

## DPO-300 自动评估与固定 prompt 对比

2026-06-03 已完成 DPO-300 在原 `test_100` 上的自动评估。

两段式评估命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval_dpo_300.yaml --skip_perplexity --max_new_tokens 128 --predictions_file results\predictions_qwen05b_dpo_300.jsonl --metrics_file results\metrics_qwen05b_dpo_300.json

D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval_dpo_300.yaml --skip_generation --predictions_file results\predictions_qwen05b_dpo_300.jsonl --metrics_file results\metrics_qwen05b_dpo_300.json
```

结果：

| 指标 | 结果 |
|---|---:|
| examples | 100 |
| ROUGE-L | 0.2436 |
| eval loss | 1.8466 |
| perplexity | 6.3381 |

对比 SFT-3k 在同一 `test_100` 上的 ROUGE-L 约 0.2539、eval loss 约 1.8221、perplexity 约 6.1851，DPO-300 在原 SFT 自动指标上没有更优。因此后续结论必须保守，不能写成 DPO 全面提升；更适合用 preference accuracy 回答 DPO 是否更偏向偏好数据中的 chosen。

固定 prompt 对比：

| 文件 | 状态 |
|---|---|
| `results/manual_compare_sft_dpo_qwen05b.jsonl` | 已生成 30 行，`sft_response` / `dpo_response` 均非空，不提交 |

## Preference Accuracy 结果

2026-06-03 已完成 SFT vs DPO 在 `preference_test_100` 上的 chosen/rejected logprob 偏好比较。

命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate_preference_accuracy.py --config configs\dpo_qwen_0.5b.yaml --output_file results\preference_accuracy_sft_dpo_qwen05b.json --records_file results\preference_accuracy_sft_dpo_qwen05b_records.jsonl
```

结果：

| 模型 | chosen preferred | rejected preferred | accuracy | mean margin |
|---|---:|---:|---:|---:|
| SFT-3k adapter | 7 / 100 | 93 / 100 | 0.07 | -0.7120 |
| DPO-300 adapter | 13 / 100 | 87 / 100 | 0.13 | -0.5464 |

结论：DPO-300 相比 SFT-3k 更偏向 chosen，但绝对 preference accuracy 仍低，不能夸大为偏好对齐充分成功。

## 当前命令

生成偏好 prompt 池：

```powershell
python -B scripts\prepare_preference_data.py --config configs\dpo_qwen_0.5b.yaml --stage prompts
```

用 SFT adapter 为偏好 prompt 生成 rejected responses：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\preference_prompts_pool.jsonl --output_file results\preference_rejected_qwen05b_qlora_3k.jsonl --mode adapter --max_new_tokens 128
```

将 rejected prediction 文件转成 DPO chosen/rejected 数据：

```powershell
python -B scripts\prepare_preference_data.py --config configs\dpo_qwen_0.5b.yaml --stage pairs
```

## 计划产出

| 文件 | 用途 |
|---|---|
| `data/processed/preference_train_50.jsonl` | DPO smoke test |
| `data/processed/preference_train_300.jsonl` | DPO 第一版主实验 |
| `data/processed/preference_valid_100.jsonl` | DPO 验证集 |
| `data/processed/preference_test_100.jsonl` | DPO 偏好评估集 |

训练集关系：

```text
preference_train_50 是 preference_train_300 的子集
preference_valid_100 / preference_test_100 与 train_300 不重叠
```

## 下一步

1. 做最终文档审查，确认 README、`docs/progress.md`、`docs/dpo_alignment_results.md` 的结论一致且表达保守。
2. 做提交边界审查，确认不提交 `models/`、`outputs/`、`data/processed/` 和完整 `results/*.json/jsonl`。
3. 如需要继续扩展，再考虑更高质量 rejected 构造或更大偏好数据，但不要抢占当前最小闭环收尾。
