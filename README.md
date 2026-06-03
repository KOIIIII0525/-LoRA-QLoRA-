# Chinese Instruction Tuning and Preference Alignment

基于 `Qwen2.5-0.5B-Instruct` 和 BELLE 中文指令数据子集的轻量 post-training 实验项目。项目在 RTX 3060 Laptop 6GB 显存约束下，完成 QLoRA 指令微调、小规模 DPO 偏好对齐、推理对比、自动评估和人工样例分析。

项目关注的是低资源环境下的完整实验流程，而不是大规模训练。当前主线如下：

```text
数据处理 -> QLoRA SFT -> 偏好数据构造 -> DPO 对齐 -> 推理对比 -> 自动评估 -> 人工分析
```

## 功能

- 构造 `train_100 / train_1k / train_3k / valid_300 / test_100 / manual_prompts_30` 小规模 SFT 数据划分。
- 在 6GB 显存笔记本 GPU 上跑通 `Qwen2.5-0.5B-Instruct` QLoRA 指令微调。
- 提供训练、推理、评估脚本，支持 dry-run、环境检查和批量推理。
- 在固定测试集上计算 ROUGE-L、eval loss 和 perplexity。
- 使用固定 prompt 对比 Base / SFT / DPO 输出，并记录成功样例和失败样例。
- 构造 `chosen/rejected` 偏好数据，完成 DPO-50 smoke test 和 DPO-300 主实验。
- 使用 `.gitignore` 排除原始数据、处理后数据、模型权重、训练输出和完整本地结果。

## 当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第 0 阶段 | 环境确认、仓库保护、`.gitignore` | 已完成 |
| 第 1 阶段 | 项目结构、配置、脚本和工具函数 | 已完成 |
| 第 2 阶段 | 构造小规模 SFT 数据集 | 已完成 |
| 第 3 阶段 | `train_100` QLoRA smoke test | 已完成 |
| 第 4 阶段 | `train_1k` / `train_3k` SFT 实验 | 已完成 |
| 第 5 阶段 | 推理、评估、固定 prompt 对比、人工分析 | 已完成 |
| 第 6 阶段 | 复现命令、结果摘要和版本管理边界整理 | 已完成 |
| 第 7 阶段 | 终端推理确认、rank 消融预留 | 已轻量收束 |
| 第 8 阶段 | DPO 偏好数据构造 | 已完成 |
| 第 9 阶段 | DPO 对齐训练 | 已完成 DPO-50 smoke 和 DPO-300 主实验 |
| 第 10 阶段 | Base / SFT / DPO 三模型评估 | 已完成核心指标和 fixed prompt 人工分析 |

详细进度见 [docs/progress.md](docs/progress.md)。

## 实验配置

| 项目 | 设置 |
|---|---|
| base model | `Qwen2.5-0.5B-Instruct` |
| 本地模型路径 | `models/qwen2.5-0.5b-instruct/` |
| tuning method | QLoRA / LoRA adapter |
| quantization | 4-bit |
| train split | `train_100`、`train_1k`、`train_3k` |
| eval split | `valid_300` |
| test split | `test_100` |
| max seq len | 512 |
| batch size | 1 |
| gradient accumulation | 8 |
| LoRA r / alpha | 8 / 16 |
| SFT precision | fp16 |
| DPO precision | fp16 关闭 |
| hardware | RTX 3060 Laptop 6GB |

完整配置见 [configs/lora_qwen_0.5b.yaml](configs/lora_qwen_0.5b.yaml)、[configs/eval.yaml](configs/eval.yaml) 和 [configs/dpo_qwen_0.5b.yaml](configs/dpo_qwen_0.5b.yaml)。

## 实验结果

### QLoRA SFT

训练阶段结果：

| 实验 | 训练数据 | step | 最终 eval loss | 本地输出目录 |
|---|---:|---:|---:|---|
| QLoRA-smoke | 100 | 13 / 13 | 约 1.9051 | `outputs/smoke/` |
| QLoRA-1k | 1000 | 125 / 125 | 约 1.7662 | `outputs/qwen05b_qlora_1k/` |
| QLoRA-3k | 3000 | 375 / 375 | 约 1.7448 | `outputs/qwen05b_qlora_3k/` |

自动评估结果：

| 指标 | 数据集 | 结果 |
|---|---|---:|
| ROUGE-L | `test_100.jsonl` | 约 0.2539 |
| eval loss | `test_100.jsonl` | 约 1.8221 |
| perplexity | `test_100.jsonl` | 约 6.1851 |

人工分析摘要：

| 维度 | 观察 |
|---|---|
| 指令遵循 | Adapter 在日期抽取、方案生成等任务上更倾向于直接回答问题 |
| 结构化输出 | Adapter 更常输出编号列表和短段落 |
| 回答完整性 | 长文、代码和多步骤任务仍可能被生成长度截断 |
| 中文表达 | 多数输出通顺，但内容深度和事实可靠性不稳定 |
| 失败案例 | 太阳系概括、Touch ID 操作指导、LoRA 概念解释等样例暴露事实错误或概念混淆 |

阶段结论：在 6GB 消费级显卡约束下，QLoRA SFT 流程能够完整运行。小规模 SFT 后，模型在部分固定 prompt 上表现出更强的结构化输出和任务导向回答倾向；但仍存在事实错误、概念混淆和生成截断，因此只能表述为观察到部分改善。

详细结果见 [docs/experiment_results.md](docs/experiment_results.md)，少量样例摘录见 [docs/result_samples.md](docs/result_samples.md)。

### DPO 偏好对齐

DPO 阶段使用小规模离线偏好数据：

| 字段 | 来源 |
|---|---|
| `prompt` | 从 SFT 训练集外样本抽取 |
| `chosen` | BELLE 原始参考答案 |
| `rejected` | QLoRA-3k SFT adapter 生成答案 |

当前已完成：

| 产物 | 状态 |
|---|---|
| `configs/dpo_qwen_0.5b.yaml` | 已新增 |
| `scripts/prepare_preference_data.py` | 已新增 |
| `scripts/train_dpo.py` | 已新增 |
| `src/preference_utils.py` | 已新增 |
| `src/dpo_utils.py` | 已新增 |
| `data/processed/preference_train_50.jsonl` | 本地已生成，不纳入版本管理 |
| `data/processed/preference_train_300.jsonl` | 本地已生成，不纳入版本管理 |
| `data/processed/preference_valid_100.jsonl` | 本地已生成，不纳入版本管理 |
| `data/processed/preference_test_100.jsonl` | 本地已生成，不纳入版本管理 |
| `outputs/qwen05b_dpo_50/` | 本地已完成 DPO smoke adapter，不纳入版本管理 |
| `outputs/qwen05b_dpo_300/` | 本地已完成 DPO 主实验 adapter，不纳入版本管理 |

DPO-300 训练结果：

| 指标 | 结果 |
|---|---:|
| train rows | 300 |
| valid rows | 100 |
| optimizer steps | 38 |
| train loss | 约 0.419 |
| eval loss | 约 0.3049 |
| eval rewards/accuracies | 约 0.93 |

三模型评估：

| 评估项 | 结果 |
|---|---|
| SFT-3k `test_100` 自动评估 | ROUGE-L 约 0.2539，eval loss 约 1.8221，ppl 约 6.1851 |
| DPO-300 `test_100` 自动评估 | ROUGE-L 约 0.2436，eval loss 约 1.8466，ppl 约 6.3381 |
| preference accuracy | SFT：0.07；DPO：0.13 |

DPO 在 preference accuracy 上相对更偏向 chosen，但绝对准确率仍低；同时 DPO 在原 SFT 测试集上的自动指标没有优于 SFT。因此当前结论应保守描述为：DPO 对输出风格和偏好排序有局部影响，但不能说明通用问答质量提升。

详细规划见 [docs/dpo_alignment_plan.md](docs/dpo_alignment_plan.md)，DPO 结果记录见 [docs/dpo_alignment_results.md](docs/dpo_alignment_results.md)。

## 快速开始

### 1. 安装依赖

推荐使用 Python 3.10 或 3.11。当前本地训练使用 `pytorch_env`：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -m pip install -r requirements.txt
```

环境细节和模型下载说明见 [docs/setup.md](docs/setup.md)。

### 2. 准备数据

仓库不包含原始 BELLE 数据和生成后的 `data/processed/`。本地需要先准备配置中的源文件：

```text
data/BelleGroup_sft.jsonl
```

生成小规模 SFT 数据集：

```powershell
python -B scripts\prepare_sft_data.py --config configs\lora_qwen_0.5b.yaml
```

### 3. 训练前检查

轻量 dry-run，不加载模型：

```powershell
python -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --dry_run
```

检查训练环境和 CUDA：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --check_env
```

### 4. SFT Smoke Test

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --output_dir outputs\smoke
```

### 5. SFT 主实验

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_3k --output_dir outputs\qwen05b_qlora_3k
```

### 6. 推理

单条 adapter 推理：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --prompt "请解释什么是 LoRA" --mode adapter
```

固定 prompt 批量对比：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\manual_prompts_30.jsonl --output_file results\manual_compare_qwen05b_qlora_3k.jsonl --mode both --max_new_tokens 128
```

### 7. 自动评估

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval.yaml
```

复用已有预测文件补指标：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval.yaml --skip_generation --predictions_file results\predictions_qwen05b_qlora_3k.jsonl --metrics_file results\metrics_qwen05b_qlora_3k.json
```

### 8. DPO 数据和训练

生成 preference prompt 池：

```powershell
python -B scripts\prepare_preference_data.py --config configs\dpo_qwen_0.5b.yaml --stage prompts
```

用 SFT adapter 分批生成 rejected response 后，构造 DPO split：

```powershell
python -B scripts\prepare_preference_data.py --config configs\dpo_qwen_0.5b.yaml --stage pairs
```

DPO dry-run：

```powershell
python -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_300 --dry_run
```

DPO 主实验：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_300 --output_dir outputs\qwen05b_dpo_300
```

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── lora_qwen_0.5b.yaml
│   ├── lora_qwen_0.5b_rank4.yaml
│   ├── dpo_qwen_0.5b.yaml
│   ├── eval.yaml
│   ├── eval_dpo_50.yaml
│   ├── eval_dpo_300.yaml
│   ├── eval_rank4.yaml
│   └── eval_rank4_1k.yaml
├── scripts/
│   ├── prepare_sft_data.py
│   ├── prepare_preference_data.py
│   ├── train_lora.py
│   ├── train_dpo.py
│   ├── infer.py
│   ├── evaluate.py
│   └── evaluate_preference_accuracy.py
├── src/
│   ├── data_utils.py
│   ├── train_utils.py
│   ├── dpo_utils.py
│   ├── eval_utils.py
│   ├── preference_utils.py
│   └── preference_eval_utils.py
├── tests/
├── docs/
│   ├── environment.md
│   ├── setup.md
│   ├── progress.md
│   ├── experiment_results.md
│   ├── dpo_alignment_plan.md
│   ├── dpo_alignment_results.md
│   ├── rank4_1k_ablation.md
│   ├── result_samples.md
│   └── error_log.md
├── data/
│   └── samples/
├── results/
└── assets/
```

## Git 与本地产物策略

以下内容只保留在本地，不纳入版本管理：

- 原始数据：`data/BelleGroup/`、`data/BelleGroup_sft*.jsonl`、SeqMonkey 原始文件。
- 处理后数据：`data/processed/`。
- 本地模型和权重：`models/`、`*.safetensors`、`*.pth`、`*.pt`、`*.bin`。
- 训练输出：`outputs/`。
- 完整本地评估结果：`results/*.jsonl`、`results/*.json`。

[docs/result_samples.md](docs/result_samples.md) 保存少量指标和样例摘录，用于在不加入完整结果文件的情况下查看实验现象。

## 项目边界

- 不从零预训练大模型。
- 不把 1.5B 模型作为当前主线。
- 不提交原始大数据、模型权重和训练输出。
- 当前版本优先保证低显存可运行、流程可复现、结论有实验依据。

## 可选扩展

- 3k vs 5k 数据量对比。
- LoRA rank 4 vs 8 消融。
- LoRA vs QLoRA 显存与效果对比。
- Web UI 推理入口。
- TinyLlama 或 Qwen2.5-1.5B 作为后续对比。

## 致谢

本项目基于本地 Tiny LLM 学习仓库改造，并使用 BELLE 中文指令数据子集进行实验。数据和模型仅用于学习与研究。
