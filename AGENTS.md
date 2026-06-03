# Agent Instructions

## 项目目标

本项目是一个基于轻量开源模型的中文指令微调与偏好对齐实验。当前主线为：

```text
数据处理 -> QLoRA SFT -> 推理评估 -> 偏好数据构造 -> DPO 对齐 -> Base/SFT/DPO 对比
```

项目重点不是大规模训练，而是把低资源环境下的 LLM post-training 流程跑通、记录清楚，并保留可复现的配置、脚本和结果摘要。

## 当前状态

项目已完成 QLoRA/SFT 第一版闭环，并扩展到基于 Qwen2.5-0.5B 的中文指令微调与偏好对齐实验。

当前已完成：

- 数据子集构造：`train_100`、`train_1k`、`train_3k`、`valid_300`、`test_100`、`manual_prompts_30`。
- `Qwen/Qwen2.5-0.5B-Instruct` 本地模型下载。
- `train_100` QLoRA smoke test。
- `train_1k` / `train_3k` SFT 主实验。
- 推理脚本、自动评估、固定 prompt 对比和人工分析。
- DPO 偏好数据构造。
- DPO-50 smoke test 和 DPO-300 主实验。
- DPO 自动评估、SFT vs DPO fixed prompt 对比、preference accuracy 和人工分析。

除非明确要求，不要继续扩大 SFT 训练规模，也不要把 PPO 或完整 RLHF 作为当前主线。

## 硬件约束

本地机器为 RTX 3060 Laptop 6GB。

默认实验设计应遵循：

- 优先使用轻量模型，例如 Qwen2.5-0.5B。
- 当前主线不使用 1.5B 模型。
- 优先考虑 QLoRA；如果 Windows 环境不稳定，可迁移到 WSL2 或 Colab。
- 默认 `max_seq_len` 从 512 开始。
- 默认 `per_device_train_batch_size=1`。
- 使用梯度累积和 gradient checkpointing 控制显存。

## 依赖策略

推荐环境：

- OS：Windows + WSL2 Ubuntu 优先；Windows 原生可用于数据处理和普通 LoRA 调试。
- Python：3.10 或 3.11。
- GPU：RTX 3060 Laptop 6GB。
- CUDA / PyTorch：以本机 CUDA 兼容版本为准，优先安装官方 PyTorch CUDA 版本。

MVP 必需依赖：

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `peft`
- `sentencepiece`
- `protobuf`
- `numpy`
- `tqdm`
- `pyyaml`
- `evaluate`
- `rouge-score`

QLoRA / 低显存相关依赖：

- `bitsandbytes`
- `scipy`

DPO 相关依赖：

- `trl`

可选依赖：

- `matplotlib`
- `pandas`
- `gradio`

依赖策略：

- 第一版先保证 `Qwen2.5-0.5B + LoRA/QLoRA + 3k 数据` 跑通。
- 不为了兼容过多环境引入复杂依赖。
- 如果 QLoRA 在 Windows 原生环境失败，优先切换到 WSL2/Colab，而不是反复修改核心训练逻辑。

## 数据规模

不要使用全量 BELLE 或几十 GB 原始数据作为当前主线。

推荐数据规模：

- `train_100.jsonl`：smoke test，验证流程跑通。
- `train_1k.jsonl`：快速实验。
- `train_3k.jsonl`：第一版主实验。
- `valid_300.jsonl`：验证集。
- `test_100.jsonl`：自动评估集。
- `manual_prompts_30.jsonl`：人工对比 prompt。

如果 3k 实验稳定，再扩展到 5k 作为后续实验。

## 实验设计

第一版实验矩阵：

| 实验 | 数据量 | 目的 |
|---|---:|---|
| Base Model | 0 | 未微调基线 |
| LoRA/QLoRA-1k | 1k | 验证小数据微调效果 |
| LoRA/QLoRA-3k | 3k | 主实验结果 |

可选扩展：

- 3k vs 5k 数据量对比。
- LoRA rank 4 vs 8 对比。
- LoRA vs QLoRA 显存和效果对比。

## 评估闭环

项目必须体现评估闭环，不能只记录“训练成功”。

至少包含：

1. 训练指标：train loss、eval loss、perplexity。
2. 自动指标：ROUGE-L 或 BERTScore，并说明其局限。
3. 固定 prompt 对比：Base Model vs LoRA/QLoRA Model。
4. 人工分析维度：
   - 指令遵循
   - 回答完整性
   - 中文表达
   - 结构化输出能力
   - 幻觉或答非所问情况

## 表达原则

README 和 docs 应像正常项目文档，不写个人申请或个人包装导向的表述。实验结论保持保守：

- 优先说“在小规模验证集和固定 prompt 上观察到改善”。
- 不夸大为模型能力全面提升。
- DPO 阶段称为“小规模离线偏好对齐 / RLAIF 风格实验”。
- 不声称完成完整 RLHF。

推荐项目简介：

> 基于 Qwen2.5 等轻量模型，完成中文指令数据处理、LoRA/QLoRA 微调、自动评估、本地推理和小规模 DPO 偏好对齐的端到端实验项目。

## 阶段路线

| 阶段 | 目标 | 产出 | 当前状态 |
|---|---|---|---|
| 第 0 阶段 | 明确环境和保护仓库 | `.gitignore`、环境记录 | 已完成 |
| 第 1 阶段 | 项目结构初始化 | `README.md`、`configs/`、`scripts/`、`src/` | 已完成 |
| 第 2 阶段 | 构造小数据集 | `train_100/1k/3k`、`valid_300`、`test_100` | 已完成 |
| 第 3 阶段 | 跑通 smoke test | 100 条数据训练成功 | 已完成 |
| 第 4 阶段 | 跑主实验 | 1k、3k LoRA/QLoRA 实验 | 已完成 |
| 第 5 阶段 | 做评估闭环 | loss、ppl、ROUGE-L、固定 prompt 对比 | 已完成 |
| 第 6 阶段 | 版本整理 | README、结果表、可复现命令 | 已完成 |
| 第 7 阶段 | 扩展实验预留 | 终端推理、rank 消融预留 | 已轻量收束 |
| 第 8 阶段 | 偏好数据构造 | preference prompt pool、chosen/rejected 数据 | 已完成 |
| 第 9 阶段 | DPO 对齐训练 | `preference_train_50`、`preference_train_300` | 已完成 |
| 第 10 阶段 | 三模型对比评估 | Base / SFT / DPO 自动指标、fixed prompt、preference accuracy | 已完成核心闭环 |

## DPO / 偏好对齐原则

第一版 DPO 不做完整人类 RLHF，也不把 PPO 作为主线。

推荐路线：

```text
chosen = BELLE 原始参考答案
rejected = QLoRA-3k SFT adapter 生成答案
```

偏好数据格式：

```json
{
  "id": "pref_0001",
  "prompt": "用户指令",
  "chosen": "参考答案",
  "rejected": "SFT adapter 生成答案",
  "source": "reference_vs_adapter_response"
}
```

DPO 阶段已完成：

- `configs/dpo_qwen_0.5b.yaml`
- `scripts/prepare_preference_data.py`
- `scripts/train_dpo.py`
- `src/preference_utils.py`
- `src/dpo_utils.py`
- `src/preference_eval_utils.py`
- `scripts/evaluate_preference_accuracy.py`
- `docs/dpo_alignment_plan.md`
- `docs/dpo_alignment_results.md`

本地已生成但不纳入版本管理：

- `data/processed/preference_prompts_pool.jsonl`
- `results/preference_rejected_*.jsonl`
- `data/processed/preference_train_50.jsonl`
- `data/processed/preference_train_300.jsonl`
- `data/processed/preference_valid_100.jsonl`
- `data/processed/preference_test_100.jsonl`
- `outputs/qwen05b_dpo_50/`
- `outputs/qwen05b_dpo_300/`

DPO 阶段 `fp16=false`，因为当前 TRL/Transformers/Accelerate/PyTorch 组合下 `fp16=true` 会触发 bfloat16 GradScaler 错误。

DPO-300 关键结果：

- 38 step。
- train loss 约 `0.419`。
- eval loss 约 `0.3049`。
- eval rewards/accuracies 约 `0.93`。
- DPO-300 在 `test_100` 上 ROUGE-L 约 `0.2436`，eval loss 约 `1.8466`，perplexity 约 `6.3381`。
- preference accuracy：SFT 为 `0.07`，DPO 为 `0.13`。

结论应保持保守：DPO 相对更偏向 chosen，但绝对准确率仍低。

## Git 与文件边界

不要提交：

- 完整 `results/*.json/jsonl`
- `data/processed/`
- `outputs/`
- `models/`
- 模型权重和 checkpoint
- 原始大数据
- 缓存文件和临时文件

可以提交：

- 源码：`scripts/`、`src/`、`tests/`
- 配置：`configs/`
- 文档：`README.md`、`docs/`
- 轻量占位文件：`assets/.gitkeep`、`data/samples/.gitkeep`、`results/.gitkeep`

## 任务管理

后续推进时，优先把任务拆成 P0 / P1 / P2：

- P0：决定项目能否跑通和复现的任务。
- P1：明显提升项目质量和可读性的任务。
- P2：后续加分项，不影响当前版本交付。

每次开始一个新任务前，应先明确：

1. 任务目标是什么。
2. 会改哪些文件。
3. 预期产出是什么。
4. 如何验证完成。

除非用户明确要求，不要同时推进太多方向。优先保证一个最小闭环完整完成。

## 审查规则

后续如用户要求 review、审查、检查、把关，应默认进入代码审查视角。

审查重点按优先级排序：

1. 是否能在 6GB 显存约束下实际运行。
2. 数据路径、模型路径、输出路径是否清晰可配置。
3. 是否存在大文件、权重、缓存误入 Git 的风险。
4. 训练、评估、推理流程是否能独立复现。
5. 是否有 baseline、验证集和测试集，避免只保留单次生成效果。
6. 代码是否过度复杂，是否偏离“能跑通、能复现、能讲清楚”的目标。
7. README 和 docs 描述是否真实可信，避免夸大。

审查输出建议使用：

- P0：必须修，否则项目无法稳定跑通或容易误导读者。
- P1：建议修，会明显提升项目质量。
- P2：可选优化，作为后续扩展项。

## 文档同步

当完成阶段性节点、遇到代表性环境/训练/评估问题，或改变项目状态时，应同步更新相关文档：

- `docs/progress.md`：记录阶段进展、当前状态、下一步。
- `docs/error_log.md`：只记录有复盘价值的典型错误。
- `README.md`：只在项目定位、运行方式、实验结果或版本管理边界发生变化时更新。
- `AGENTS.md`：只记录长期协作规则、项目原则和阶段路线，不展开临时错误细节。
