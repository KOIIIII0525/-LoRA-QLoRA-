# 第 0 阶段：运行环境与数据策略

## 当前本地情况

- 工作目录：`D:\100programs\LLaMA2_model`
- 系统环境：Windows / PowerShell
- 当前默认 Python：`3.13.9`
- 已发现 Conda 环境：`D:\anaconda3\envs\pytorch_env`
- `pytorch_env` Python：`3.12.12`
- `pytorch_env` PyTorch：`2.10.0+cu126`
- `pytorch_env` CUDA available：True
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU
- 显存：6GB
- NVIDIA Driver：`581.29`
- `nvidia-smi` 显示 CUDA Version：`13.0`

注意：当前默认 Python `3.13.9` 不建议直接作为训练环境。已有 `pytorch_env` 已补齐 LoRA/QLoRA smoke test 所需依赖，可以作为第一版优先训练环境。

额外注意：原旧目录 `code/` 曾遮蔽 Python 标准库 `code` 模块，导致在项目根目录直接 `import torch` 失败。该目录已重命名为 `tiny_llm_legacy/`，目前不再影响 PyTorch 导入。

## 第一版推荐运行环境

第一版优先目标是跑通小规模 LoRA/QLoRA 中文指令微调闭环。

推荐环境：

- Python：优先使用已有 `pytorch_env` 做 smoke test；如果依赖冲突明显，再新建 `3.10` 或 `3.11` 环境
- GPU：RTX 3060 Laptop 6GB
- 模型：`Qwen2.5-0.5B-Instruct`
- 训练方式：优先 QLoRA；如果 Windows 原生环境不稳定，切换到 WSL2 Ubuntu 或 Colab
- `max_seq_len`：512
- `per_device_train_batch_size`：1
- `gradient_accumulation_steps`：8 或 16
- LoRA rank：8
- LoRA alpha：16
- precision：fp16
- gradient checkpointing：开启

环境选择建议：

1. 数据处理可以先在 Windows 原生环境完成。
2. 训练优先使用 WSL2 Ubuntu 或 Colab。
3. 如果坚持 Windows 原生训练，先跑普通 LoRA；QLoRA 的 `bitsandbytes` 兼容性需要单独验证。

## `pytorch_env` 当前检查结果

在项目目录外检查：

- Python：`3.12.12`
- PyTorch：`2.10.0+cu126`
- CUDA available：True
- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`

在项目根目录运行训练环境检查：

```bash
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --check_env
```

当前结果：

- PyTorch 环境本身可用。
- 旧 `code/` 目录已重命名为 `tiny_llm_legacy/`，不再遮蔽标准库。
- 已安装 `accelerate`、`peft`、`bitsandbytes`、`sentencepiece`、`evaluate`、`rouge-score`。
- `scripts/train_lora.py --check_env` 已通过。

## 当前数据文件判断

当前目录已有以下数据：

| 文件 | 大小/行数 | 处理策略 |
|---|---:|---|
| `data/mobvoi_seq_monkey_general_open_corpus.jsonl` | 约 33GB | 第一版不用 |
| `data/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2` | 约 11GB | 第一版不用 |
| `data/BelleGroup/train_3.5M_CN.json` | 约 4.8GB | 第一版不用 |
| `data/BelleGroup_sft.jsonl` | 10001 行 | 用作抽样来源 |
| `data/BelleGroup_sft_1k.jsonl` | 1000 行 | 用作 smoke test / 1k 实验来源 |
| `seq_monkey_datawhale.jsonl` | 0 字节 | 不使用 |

## 第一版数据策略

不删除现有大数据，也不从全量原始 BELLE 重新处理。

第一版直接从已处理好的 `data/BelleGroup_sft.jsonl` 和 `data/BelleGroup_sft_1k.jsonl` 生成轻量子集：

- `data/processed/train_100.jsonl`
- `data/processed/train_1k.jsonl`
- `data/processed/train_3k.jsonl`
- `data/processed/valid_300.jsonl`
- `data/processed/test_100.jsonl`
- `data/processed/manual_prompts_30.jsonl`

推荐抽样方式：

- 固定随机种子：`seed=42`
- 从 `data/BelleGroup_sft.jsonl` 中抽样和划分
- `train_100` 用于 smoke test
- `train_1k` 用于快速实验
- `train_3k` 用于第一版主实验
- `valid_300` 用于 eval loss / perplexity
- `test_100` 用于自动指标
- `manual_prompts_30` 用于固定人工对比

## 第 0 阶段结论

第一版项目路线确定为：

> 在 RTX 3060 Laptop 6GB 约束下，基于已处理的 BELLE 中文指令数据子集，使用 Qwen2.5-0.5B-Instruct 做 LoRA/QLoRA 微调，先完成 100 条 smoke test，再完成 1k 和 3k 实验，并建立评估闭环。
