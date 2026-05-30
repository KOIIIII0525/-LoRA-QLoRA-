# Lightweight Chinese Instruction Tuning Lab

基于 `Qwen2.5-0.5B-Instruct` 和 BELLE 中文指令数据子集，在 RTX 3060 Laptop 6GB 显存约束下完成 LoRA / QLoRA 指令微调、推理对比、自动评估和人工分析的端到端实验项目。

本项目定位为适合“小厂大模型算法实习”投递和面试讲解的个人项目。目标不是追求大规模训练或工业级 SOTA，而是把 LLM 微调的核心流程跑通、复现清楚、结论讲得保守可信：

```text
数据处理 -> QLoRA 训练 -> 推理对比 -> 自动评估 -> 人工分析 -> GitHub 展示
```

## 项目亮点

- 在消费级 6GB 显存笔记本 GPU 上跑通 `Qwen2.5-0.5B-Instruct` QLoRA 微调。
- 构造 `train_100 / train_1k / train_3k / valid_300 / test_100 / manual_prompts_30` 小规模实验闭环。
- 完成 `train_100` smoke test、`train_1k` 快速实验和 `train_3k` 第一版主实验。
- 提供训练、推理、评估脚本，支持 dry-run 检查，便于复现路径和环境问题定位。
- 在 `test_100` 上保存预测并计算 ROUGE-L、eval loss、perplexity。
- 在固定 prompt 上对比 Base Model 与 QLoRA Adapter，并保留成功样例和失败样例。
- 使用 `.gitignore` 排除大数据、模型权重、训练输出和完整本地结果，GitHub 只提交代码、配置、文档和小样例摘要。

## 当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第 0 阶段 | 环境确认、仓库保护、`.gitignore` | 已完成 |
| 第 1 阶段 | 项目结构、README、configs、scripts、src | 已完成 |
| 第 2 阶段 | 构造小规模 SFT 数据集 | 已完成 |
| 第 3 阶段 | `train_100` QLoRA smoke test | 已完成 |
| 第 4 阶段 | `train_1k` / `train_3k` 主实验 | 已完成 |
| 第 5 阶段 | 推理、评估、固定 prompt 对比、人工分析 | 已完成 |
| 第 6 阶段 | GitHub 展示材料整理与首版提交 | 已完成 |
| 第 7 阶段 | 加分项：展示微调、Demo 或扩展实验 | 可选，待开始 |

详细进度见 [docs/progress.md](docs/progress.md)。

## 实验配置

| 项目 | 第一版设置 |
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
| precision | fp16 |
| hardware | RTX 3060 Laptop 6GB |

完整配置见 [configs/lora_qwen_0.5b.yaml](configs/lora_qwen_0.5b.yaml) 和 [configs/eval.yaml](configs/eval.yaml)。

## 实验结果

训练阶段结果：

| 实验 | 训练数据 | step | 最终 eval loss | 输出目录 |
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

第一版结论应保守表述为：

> 在 6GB 消费级显卡约束下，本项目完成了 Qwen2.5-0.5B 的 QLoRA 中文指令微调和评估闭环。小规模 SFT 后，模型在部分固定 prompt 上表现出更强的结构化输出和任务导向回答倾向；但仍存在事实错误、概念混淆和生成截断，因此只能说观察到部分改善，不能夸大为能力显著提升。

详细结果见 [docs/experiment_results.md](docs/experiment_results.md)，可提交的样例摘录见 [docs/result_samples.md](docs/result_samples.md)。

## 快速开始

### 1. 安装依赖

推荐使用已验证的本机 `pytorch_env`，默认系统 Python 只用于轻量脚本检查。

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -m pip install -r requirements.txt
```

环境细节和模型下载说明见 [docs/setup.md](docs/setup.md)。

### 2. 准备数据

本仓库不提交大规模原始数据和生成后的 `data/processed/`。本地需要先准备配置中的源文件：

```text
data/BelleGroup_sft.jsonl
```

生成小规模数据集：

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

### 4. Smoke Test

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --output_dir outputs\smoke
```

### 5. 主实验

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

## 项目结构

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── lora_qwen_0.5b.yaml
│   └── eval.yaml
├── scripts/
│   ├── prepare_sft_data.py
│   ├── train_lora.py
│   ├── infer.py
│   └── evaluate.py
├── src/
│   ├── data_utils.py
│   ├── train_utils.py
│   └── eval_utils.py
├── tests/
├── docs/
│   ├── environment.md
│   ├── setup.md
│   ├── progress.md
│   ├── experiment_results.md
│   ├── result_samples.md
│   └── error_log.md
├── data/
│   └── samples/
├── results/
└── assets/
```

## Git 与大文件策略

以下内容只保留在本地，不提交到 Git：

- 原始大数据：`data/BelleGroup/`、`data/BelleGroup_sft*.jsonl`、SeqMonkey 原始文件。
- 处理后数据：`data/processed/`。
- 本地模型和权重：`models/`、`*.safetensors`、`*.pth`、`*.pt`、`*.bin`。
- 训练输出：`outputs/`。
- 完整本地评估结果：`results/*.jsonl`、`results/*.json`。

GitHub 展示使用 [docs/result_samples.md](docs/result_samples.md) 保存少量可提交的指标和样例摘录。

## 项目边界

- 不从零预训练大模型。
- 不追求工业级 SOTA。
- 不使用 1.5B 模型作为第一版主线。
- 不提交几十 GB 原始数据、模型权重和训练输出。
- 第一版优先保证低显存可运行、流程可复现、结论可讲清楚。

## 可选扩展

- 3k vs 5k 数据量对比。
- LoRA rank 4 vs 8 消融。
- LoRA vs QLoRA 显存与效果对比。
- Gradio Demo。
- TinyLlama 或 Qwen2.5-1.5B 作为后续对比。

## 致谢

本项目基于本地 Tiny LLM 学习仓库改造，并使用 BELLE 中文指令数据子集进行实验。数据和模型仅用于学习与研究。
