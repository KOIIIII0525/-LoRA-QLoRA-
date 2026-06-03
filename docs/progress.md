# 项目进度记录

## 项目一句话简介

基于 Qwen2.5-0.5B 的中文指令微调与偏好对齐实验项目。

## 阶段总览

核心目标只有一个：

```text
先完整跑通 数据处理 -> QLoRA SFT -> 推理评估 -> 偏好数据构造 -> DPO 对齐 -> Base/SFT/DPO 对比 的最小闭环。
```

| 阶段 | 目标 | 产出 | 是否必须 | 当前状态 |
|---|---|---|---|---|
| 第 0 阶段 | 明确环境和保护仓库 | `.gitignore`、环境记录 | 必须 | 已完成 |
| 第 1 阶段 | 项目结构初始化 | `README.md`、`configs/`、`scripts/`、`src/` | 必须 | 已完成 |
| 第 2 阶段 | 构造小数据集 | `train_100/1k/3k`、`valid_300`、`test_100` | 必须 | 已完成 |
| 第 3 阶段 | 跑通 smoke test | 100 条数据训练成功 | 必须 | 已完成 |
| 第 4 阶段 | 跑主实验 | 1k、3k LoRA/QLoRA 实验 | 必须 | 已完成 |
| 第 5 阶段 | 做评估闭环 | loss、ppl、ROUGE-L、固定 prompt 对比 | 必须 | 已完成 |
| 第 6 阶段 | 版本整理 | README、结果表、可复现命令、推理说明素材 | 必须 | 已完成 |
| 第 7 阶段 | 加分实验 | 终端推理确认、rank 消融预留、后续扩展 | 可选 | 已完成轻量收束 |
| 第 8 阶段 | 偏好数据构造 | preference prompt pool、chosen/rejected 数据 | 必须 | 已完成 |
| 第 9 阶段 | DPO 对齐训练 | `preference_train_50` smoke、`preference_train_300` 主实验 | 必须 | 已完成 |
| 第 10 阶段 | 三模型对比评估 | Base / SFT / DPO 自动指标、固定 prompt、preference accuracy | 必须 | 已完成核心指标和 fixed prompt 人工分析 |

## 2026-06-03 阶段统计

### 大阶段进度

| 大阶段 | 范围 | 状态 | 关键证据 |
|---|---|---|---|
| SFT 闭环 | 第 0-7 阶段 | 已完成 | `outputs/qwen05b_qlora_3k/`、`results/metrics_qwen05b_qlora_3k.json`、`results/manual_compare_qwen05b_qlora_3k.jsonl` |
| DPO 数据与训练 | 第 8-9 阶段 | 已完成 | `preference_train_50/300`、`preference_valid_100`、`preference_test_100` 均已生成；`outputs/qwen05b_dpo_50/` 和 `outputs/qwen05b_dpo_300/` 已生成 adapter |
| 三模型评估 | 第 10 阶段 | 已完成核心指标和人工分析 | DPO-300 `test_100` 自动评估已完成；SFT vs DPO `manual_prompts_30` 对比与人工分析已完成；preference accuracy 已完成 |

### 当前关键指标

| 模型/阶段 | 数据集 | ROUGE-L | eval loss | perplexity | 说明 |
|---|---|---:|---:|---:|---|
| SFT-3k adapter | `test_100` | 0.2539 | 1.8221 | 6.1851 | 第一版 QLoRA/SFT 主实验 |
| DPO-300 adapter | `test_100` | 0.2436 | 1.8466 | 6.3381 | DPO 后在原 SFT 测试集上的自动指标略低，需结合 preference accuracy 和人工分析解释 |

| Preference accuracy | 数据集 | chosen preferred | rejected preferred | accuracy | mean margin |
|---|---|---:|---:|---:|---:|
| SFT-3k adapter | `preference_test_100` | 7 / 100 | 93 / 100 | 0.07 | -0.7120 |
| DPO-300 adapter | `preference_test_100` | 13 / 100 | 87 / 100 | 0.13 | -0.5464 |

### 当前本地产物

| 产物 | 行数/状态 | Git 策略 |
|---|---:|---|
| `results/predictions_qwen05b_dpo_300.jsonl` | 100 行 | 不提交 |
| `results/metrics_qwen05b_dpo_300.json` | 已生成 | 不提交 |
| `results/manual_compare_sft_dpo_qwen05b.jsonl` | 30 行，`sft_response` / `dpo_response` 均非空 | 不提交 |
| `results/preference_accuracy_sft_dpo_qwen05b.json` | 已生成 | 不提交 |
| `results/preference_accuracy_sft_dpo_qwen05b_records.jsonl` | 100 行 | 不提交 |

### Fixed Prompt 人工分析摘要

- DPO 输出整体更短：30 条中有 24 条比 SFT 更短，平均长度约 192.2 字符，SFT 约 306.4 字符。
- DPO 在部分任务上更贴近目标形式，例如秋天诗歌、10 句话科技趋势总结和公园景色描写。
- DPO 在日期抽取、排队等候对话、太阳系概括和 Touch ID 操作指导中出现退化或幻觉。
- 当前结论仍应保守：DPO 对输出风格和偏好排序有局部影响，但没有证明通用问答质量提升。

### 当前审查结论

- P0：未发现会阻塞当前最小闭环继续推进的问题。
- P1：阶段总览曾停留在第 7/8 阶段，已在本节补充第 8-10 阶段统计，便于后续复盘和项目交接。
- P1：DPO-300 在 `test_100` 上的 ROUGE-L、eval loss、perplexity 未优于 SFT-3k，因此结论必须保守，不能写成 DPO 全面提升。
- P1：preference accuracy 从 SFT 的 0.07 提升到 DPO 的 0.13，但绝对值仍低；应表述为“DPO 相对更偏向 chosen，但偏好排序能力仍有限”。

## 2026-06-03 README 与提交边界整理

### 本轮调整

- 重写 `README.md`，改为普通实验项目说明口吻，保留项目目标、实验配置、结果表、快速开始、项目结构和本地产物策略。
- 清理 `AGENTS.md`、`docs/experiment_results.md`、`docs/error_log.md`、`docs/progress.md`、`docs/result_samples.md` 中偏个人用途的表述，统一改为复盘、说明和版本整理口径。
- 删除 `docs/superpowers/plans/` 下的内部计划文档，避免与项目复现无关的工作流记录进入公开仓库。
- 在 `.gitignore` 中忽略 `docs/superpowers/`，防止后续自动生成的内部计划文件重新进入提交范围。

### 提交边界

本轮继续保持以下内容不纳入版本管理：

- `models/`
- `outputs/`
- `data/processed/`
- `results/*.json`
- `results/*.jsonl`
- 原始大数据和模型权重

当前可纳入版本管理的内容应聚焦在源码、配置、测试、README 和 docs 摘要文档。

## 2026-05-28

### 已完成

- 完成第 0 阶段：运行环境与数据策略确认。
- 新增 `.gitignore`，排除大数据、模型权重、缓存和训练输出。
- 新增 `docs/environment.md`，记录 RTX 3060 Laptop 6GB 下的第一版实验策略。
- 开始第 1 阶段：项目结构初始化。
- 新增 `README.md` 初稿，明确项目目标、数据策略、实验设计和评估闭环。
- 新增 `configs/lora_qwen_0.5b.yaml`，作为第一版 LoRA/QLoRA 训练配置。
- 新增 `configs/eval.yaml`，作为第一版评估配置。
- 完成第 2 阶段：数据处理脚本和小规模数据集生成。
- 新增 `src/data_utils.py`，封装 JSONL 读取、写入、样本过滤、确定性划分和 manual prompt 抽取。
- 新增 `scripts/prepare_sft_data.py`，支持从配置文件生成小规模 SFT 数据集。
- 新增 `tests/test_prepare_sft_data.py`，覆盖数据过滤、读写和确定性划分。
- 开始第 3 阶段：LoRA/QLoRA 训练入口。
- 新增 `src/train_utils.py`，封装训练配置读取、训练输入解析、数据存在性检查和 dry-run 报告。
- 新增 `scripts/train_lora.py`，支持训练前 dry-run 检查，并预留真实 LoRA/QLoRA 训练入口。
- 新增 `tests/test_train_lora.py`，覆盖训练配置读取、路径解析、样本行数检查和 dry-run 报告。
- 新增 `docs/error_log.md`，记录开发过程中可用于后续复盘的典型错误。

### 第 2 阶段数据产出

运行命令：

```bash
python -B scripts/prepare_sft_data.py --config configs/lora_qwen_0.5b.yaml
```

生成文件：

| 文件 | 行数 | 用途 |
|---|---:|---|
| `data/processed/train_100.jsonl` | 100 | smoke test |
| `data/processed/train_1k.jsonl` | 1000 | 快速实验 |
| `data/processed/train_3k.jsonl` | 3000 | 第一版主实验 |
| `data/processed/valid_300.jsonl` | 300 | eval loss / perplexity |
| `data/processed/test_100.jsonl` | 100 | 自动评估 |
| `data/processed/manual_prompts_30.jsonl` | 30 | 固定人工对比 |

验证结果：

- `python -B -m unittest discover -s tests -v`：4 个测试通过。
- `train_3k` 与 `valid_300` 重叠数：0。
- `train_3k` 与 `test_100` 重叠数：0。
- `valid_300` 与 `test_100` 重叠数：0。

### 第 3 阶段当前状态

训练前检查命令：

```bash
python -B scripts/train_lora.py --config configs/lora_qwen_0.5b.yaml --train_split train_100 --output_dir outputs/smoke --dry_run
```

dry-run 输出确认：

- base model：`models/qwen2.5-0.5b-instruct`
- QLoRA：开启
- 4-bit：开启
- train file：`data/processed/train_100.jsonl`
- eval file：`data/processed/valid_300.jsonl`
- train rows：100
- eval rows：300
- max seq len：512
- batch size：1
- gradient accumulation：8
- LoRA r：8
- LoRA alpha：16

当前验证结果：

- `python -B -m unittest discover -s tests -v`：8 个测试通过。
- `python -B -c "import scripts.train_lora; print('IMPORT_OK')"`：导入通过。
- 当前 Python 3.13 环境未执行真实训练；真实训练建议切换到 Python 3.10/3.11 深度学习环境。
- 已检查 `pytorch_env`：Python 3.12.12，PyTorch 2.10.0+cu126，CUDA 可用。
- 已在 `pytorch_env` 中安装 `accelerate`、`peft`、`bitsandbytes`、`sentencepiece`、`evaluate`、`rouge-score` 等依赖。
- 已为 `scripts/train_lora.py` 增加 `--check_env`，用于训练前检查重依赖。
- 已将旧 `code/` 目录重命名为 `tiny_llm_legacy/`，解决其遮蔽 Python 标准库 `code` 导致 PyTorch 导入失败的问题。
- `D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --check_env`：通过。
- 新增 `docs/setup.md`，集中记录环境安装、验证和 smoke test 命令。
- 已确认 Hugging Face 原站访问超时，`https://hf-mirror.com` 可访问 `Qwen/Qwen2.5-0.5B-Instruct`。
- 已通过国内镜像下载模型到 `models/qwen2.5-0.5b-instruct/`，并将训练/评估配置切换为本地模型路径。
- 已验证本地 tokenizer 可离线加载。
- 已修复当前 `transformers 5.8.0` 下的 `TrainingArguments` 和 `Trainer` API 兼容问题。
- 已完成 `train_100` QLoRA 冒烟训练，输出目录为 `outputs/smoke/`。
- 冒烟训练结果：13 个训练 step，`train_loss` 约 `2.021`，`eval_loss` 约 `1.905`，未发生 OOM。
- 已生成 LoRA adapter：`outputs/smoke/adapter_model.safetensors`。

### 当前结论

第一版不使用全量 BELLE 或 SeqMonkey 数据。项目将从 `data/BelleGroup_sft.jsonl` 中抽样生成 100 / 1k / 3k 训练集，以及验证集、测试集和人工 prompt 集。

第一版模型使用本地目录 `models/qwen2.5-0.5b-instruct/` 中的 `Qwen/Qwen2.5-0.5B-Instruct`，目标是在 6GB 显存约束下优先跑通完整闭环。

### 当时阻塞

- 真实训练冒烟测试已完成。
- 当前还缺推理脚本和评估脚本，尚未形成 Base Model vs Adapter 的固定 prompt 对比。

### 下一步

1. 进入第 4 阶段：在 smoke test 成功的配置基础上跑主实验，优先尝试 `train_1k`，稳定后跑 `train_3k`。
2. 创建推理脚本 `scripts/infer.py`，用于 Base vs Adapter 输出对比。
3. 创建评估脚本 `scripts/evaluate.py`，计算 eval loss / perplexity / ROUGE-L。
4. 在 `manual_prompts_30.jsonl` 上生成固定 prompt 对比结果。

## 2026-05-29

### 第 4 阶段拆分策略

第 4 阶段不一次性直接跑完整主实验，而是拆成三个小阶段推进，方便定位训练、显存、配置和结果解释问题：

| 子阶段 | 目标 | 预期产出 | 当前状态 |
|---|---|---|---|
| 4.1 `train_1k` 主实验 | 验证 1k 数据规模下训练稳定性，拿到比 smoke test 更有意义的训练与验证 loss | `outputs/qwen05b_qlora_1k/` adapter、checkpoint、训练日志 | 已完成 |
| 4.2 `train_3k` 主实验 | 作为第一版主结果，和 1k 对比数据量扩大后的变化 | `outputs/qwen05b_qlora_3k/` adapter、checkpoint、训练日志 | 已完成 |
| 4.3 结果整理 | 提取训练指标、产物、配置和可复盘结论，为推理和评估闭环服务 | 结果表、阶段结论、README 说明素材 | 已完成 |

这个拆分的目的不是追求一次训练跑得越大越好，而是在 6GB 显存约束下稳步扩大实验规模。`train_100` 只证明流程能跑通；`train_1k` 开始具备实验观察价值；`train_3k` 才作为第一版主结果。

### 4.1 `train_1k` 实验检查结果

产物目录：

```text
outputs/qwen05b_qlora_1k/
```

关键产物：

| 文件 | 作用 |
|---|---|
| `adapter_model.safetensors` | LoRA adapter 权重 |
| `adapter_config.json` | LoRA 配置与基座模型路径 |
| `checkpoint-100/` | 第 100 step 保存点 |
| `checkpoint-125/` | 训练完成保存点 |
| `checkpoint-125/trainer_state.json` | 训练过程指标记录 |

训练配置摘要：

| 配置项 | 值 |
|---|---|
| base model | `models/qwen2.5-0.5b-instruct` |
| train split | `data/processed/train_1k.jsonl` |
| eval split | `data/processed/valid_300.jsonl` |
| training rows | 1000 |
| eval rows | 300 |
| epoch | 1 |
| max seq len | 512 |
| batch size | 1 |
| gradient accumulation | 8 |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| QLoRA / 4-bit | 开启 |

`checkpoint-125/trainer_state.json` 显示：

| 指标 | 结果 |
|---|---:|
| `global_step` | 125 |
| `max_steps` | 125 |
| `epoch` | 1.0 |
| logged train loss 平均值 | 约 1.8355 |
| logged train loss 最小值 | 约 1.6630 |
| logged train loss 最后一次记录 | 约 1.7923 |
| step 100 `eval_loss` | 约 1.7684 |
| step 125 `eval_loss` | 约 1.7662 |

结论：

- `train_1k` 已完成完整 1 epoch，而不是中途停止。
- 训练过程中已生成最终 adapter 和两个 checkpoint。
- 相比 `train_100` smoke test 的 `eval_loss` 约 1.905，`train_1k` 最终 `eval_loss` 约 1.766，说明扩大到 1k 后在同一验证集上的 loss 有下降。
- 当前只能保守表述为“验证集 loss 下降，训练流程稳定”，还不能直接等同于真实指令遵循能力提升；后续需要第 5 阶段用固定 prompt、ROUGE-L 和人工分析补齐评估闭环。

### 4.2 `train_3k` 实验检查结果

运行目标：

```text
在 1k 实验稳定的基础上，将训练数据扩大到 3k，作为第一版主实验结果。
```

产物目录：

```text
outputs/qwen05b_qlora_3k/
```

关键产物：

| 文件 | 作用 |
|---|---|
| `adapter_model.safetensors` | 3k 主实验 LoRA adapter 权重 |
| `adapter_config.json` | LoRA 配置与基座模型路径 |
| `checkpoint-100/` | 第 100 step 保存点 |
| `checkpoint-200/` | 第 200 step 保存点 |
| `checkpoint-300/` | 第 300 step 保存点 |
| `checkpoint-375/` | 训练完成保存点 |
| `checkpoint-375/trainer_state.json` | 完整训练过程指标记录 |
| `train_stdout.log` / `train_stderr.log` | 训练输出和进度日志 |

训练配置摘要：

| 配置项 | 值 |
|---|---|
| base model | `models/qwen2.5-0.5b-instruct` |
| train split | `data/processed/train_3k.jsonl` |
| eval split | `data/processed/valid_300.jsonl` |
| training rows | 3000 |
| eval rows | 300 |
| epoch | 1 |
| max seq len | 512 |
| batch size | 1 |
| gradient accumulation | 8 |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| QLoRA / 4-bit | 开启 |

训练过程状态：

| 检查点 | 观察 |
|---|---|
| 训练进程 | `python.exe` 正常运行至结束 |
| GPU 状态 | 训练中显存接近 5.2-5.8GB，说明 6GB 显存基本打满 |
| 第一个 checkpoint | `checkpoint-100/` 正常生成 |
| 中间 checkpoint | `checkpoint-200/`、`checkpoint-300/` 正常生成 |
| 最终产物 | `checkpoint-375/` 和最终 adapter 正常生成 |

`checkpoint-375/trainer_state.json` 和训练日志显示：

| 指标 | 结果 |
|---|---:|
| `global_step` | 375 |
| `max_steps` | 375 |
| `epoch` | 1.0 |
| logged train loss 平均值 | 约 1.8104 |
| logged train loss 最小值 | 约 1.6231 |
| logged train loss 最大值 | 约 2.1582 |
| logged train loss 最后一次记录 | 约 1.7785 |
| step 100 `eval_loss` | 约 1.7682 |
| step 200 `eval_loss` | 约 1.7534 |
| step 300 `eval_loss` | 约 1.7474 |
| step 375 `eval_loss` | 约 1.7448 |
| 训练日志 `train_loss` | 约 1.811 |
| 训练总耗时 | 约 6 小时 28 分钟 |
| train samples/s | 约 0.129 |
| train steps/s | 约 0.016 |

结论：

- `train_3k` 已完成完整 1 epoch，而不是中途停止。
- 3k 主实验在 RTX 3060 Laptop 6GB + Windows 原生环境下可以跑通，但显存接近打满，训练速度较慢。
- 验证集 loss 从 step 100 的约 1.7682 逐步下降到 step 375 的约 1.7448，说明在这个小规模验证集上继续训练到 3k 数据后 loss 仍有小幅改善。
- 相比 `train_1k` 最终 `eval_loss` 约 1.7662，`train_3k` 最终 `eval_loss` 约 1.7448，观察到小幅下降。
- 这个结论仍应保守表达为“在固定验证集上的 eval loss 有小幅下降”，还不能直接说模型真实问答能力显著提升；下一步需要用固定 prompt 对比、ROUGE-L 和人工分析验证输出质量。

### 4.3 结果整理

已新增集中结果文档：

```text
docs/experiment_results.md
```

整理内容包括：

- 统一实验配置：base model、QLoRA、4-bit、max seq len、batch size、梯度累积、LoRA rank 等。
- 阶段 4 结果表：`train_100` smoke test、`train_1k`、`train_3k` 的 step、最终 eval loss、logged train loss 和产物状态。
- 3k 训练过程指标：step 100 / 200 / 300 / 375 的 eval loss 变化。
- 实验结论：验证集 loss 从 100 到 1k 明显下降，从 1k 到 3k 小幅下降。
- 结果说明口径：强调逐步扩大数据规模、定位风险、保守解释结果。
- 下一步评估闭环：推理脚本、评估脚本、固定 prompt 对比和人工分析。

阶段 4 当前可总结为：

| 实验 | 训练数据 | step | 最终 eval loss | 产物 |
|---|---:|---:|---:|---|
| QLoRA-smoke | 100 | 13 / 13 | 约 1.9051 | `outputs/smoke/` |
| QLoRA-1k | 1000 | 125 / 125 | 约 1.7662 | `outputs/qwen05b_qlora_1k/` |
| QLoRA-3k | 3000 | 375 / 375 | 约 1.7448 | `outputs/qwen05b_qlora_3k/` |

### 结果说明要点

这个阶段可以这样表达：

> 我没有一开始直接跑 3k 主实验，而是把主实验拆成 1k 和 3k 两步。先用 100 条数据做 smoke test，验证模型加载、QLoRA、显存和保存流程；再用 1k 数据验证训练是否稳定，并拿到第一组有对比意义的 loss 和 eval loss；最后再跑 3k 作为第一版主实验结果。这样做的好处是如果出现 OOM、API 不兼容或 loss 异常，可以快速定位是环境问题、配置问题还是数据规模问题。

关键概念：

- smoke test：小样本快速验证流程是否跑通，不用于证明模型效果。
- train loss：训练集上的语言建模损失，主要观察训练是否收敛、是否异常震荡。
- eval loss：验证集上的损失，用于观察模型对未参与训练样本的泛化情况。
- checkpoint：训练中间保存点，方便恢复训练或比较不同 step 的状态。
- adapter：LoRA 训练得到的增量参数，推理时需要和 base model 一起加载。
- QLoRA：用 4-bit 量化加载基座模型，只训练 LoRA adapter，降低显存占用。

### 当时阻塞

- 当前还缺推理脚本和评估脚本，尚未形成 Base Model vs Adapter 的固定 prompt 对比。
- `README.md` 还需要在评估闭环完成后补充最终结果。

### 下一步

1. 进入第 5 阶段：补齐 `scripts/infer.py` 和 `scripts/evaluate.py`。
2. 在 `manual_prompts_30.jsonl` 上生成 Base Model vs Adapter 固定 prompt 对比结果。
3. 把自动评估和人工分析结果补充到 README 与 `docs/experiment_results.md`。

## 2026-05-30

### 第 5 阶段拆分策略

第 5 阶段比训练阶段更复杂，因此拆成四个小阶段推进：

| 子阶段 | 目标 | 产出 | 当前状态 |
|---|---|---|---|
| 5.1 推理脚本 | 支持 base model、adapter、单条 prompt 和批量 prompt 推理 | `scripts/infer.py` | 已完成并通过 dry-run / 单条 adapter 推理 |
| 5.2 评估脚本 | 支持 test set 预测、ROUGE-L、可选 perplexity | `scripts/evaluate.py` | 已完成并通过 dry-run / 1 条 quick eval |
| 5.3 固定 prompt 对比 | 在 `manual_prompts_30.jsonl` 上生成 Base vs Adapter 对比 | `results/manual_compare_qwen05b_qlora_3k.jsonl` | 已完成 |
| 5.4 结果整理 | 将自动指标、固定 prompt 和人工分析写入文档 | README、`docs/experiment_results.md` | 已完成 |

### 5.1 推理脚本

新增文件：

```text
scripts/infer.py
src/eval_utils.py
tests/test_infer_script.py
tests/test_eval_utils.py
```

能力：

- `--prompt`：单条 prompt 推理。
- `--input_file`：批量读取 `manual_prompts_30.jsonl` 或其他 JSONL。
- `--mode base|adapter|both`：支持单独 base、单独 adapter 或顺序运行二者。
- `--limit`：用于小样本快速验证。
- `--dry_run`：不加载模型，只验证配置、样本数和输出路径。
- 顺序加载 base/adapter，避免同时占用两份模型显存。

已验证命令：

```powershell
python -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\manual_prompts_30.jsonl --limit 2 --mode both --dry_run
```

输出确认：

```text
INFER_DRY_RUN_OK
records: 2
output_file: results\manual_compare_qwen05b_qlora_3k.jsonl
```

真实 adapter 单条推理命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --prompt "请用三句话解释什么是 LoRA。" --mode adapter --max_new_tokens 64
```

观察到的样例问题：

- 模型把 LoRA 解释成了无线通信中的 LoRa，而不是大模型微调中的 Low-Rank Adaptation。
- 这个失败样例说明：训练 loss / eval loss 下降不等于真实指令理解和领域概念回答一定正确。
- 后续人工分析时应保留这类失败案例，用于说明项目评估是保守和可信的。

### 5.2 评估脚本

新增文件：

```text
scripts/evaluate.py
tests/test_evaluate_script.py
```

能力：

- 从 `configs/eval.yaml` 读取 base model、adapter、test set 和输出路径。
- 支持 `--limit` 做小样本快速验证。
- 支持 `--max_new_tokens` 控制 quick eval 的生成长度。
- 支持 `--skip_generation` 复用已有预测文件。
- 支持 `--skip_perplexity` 跳过较慢的 eval loss / perplexity。
- 支持 ROUGE-L 自动指标。
- 支持可选 eval loss / perplexity 计算。

已验证 dry-run：

```powershell
python -B scripts\evaluate.py --config configs\eval.yaml --limit 2 --dry_run
```

输出确认：

```text
EVAL_DRY_RUN_OK
records: 2
predictions_file: results\predictions_qwen05b_qlora_3k.jsonl
metrics_file: results\metrics_qwen05b_qlora_3k.json
```

已验证 1 条 quick eval：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval.yaml --limit 1 --skip_perplexity --max_new_tokens 64 --predictions_file results\quick_eval_predictions_1.jsonl --metrics_file results\quick_eval_metrics_1.json
```

输出指标：

```json
{
  "num_examples": 1,
  "rouge_l": {
    "count": 1,
    "rouge_l": 0.3157894736842105
  },
  "perplexity": null
}
```

### 当前验证

- `python -B -m unittest discover -s tests -v`：22 个测试通过。
- `infer.py` dry-run 通过。
- `evaluate.py` dry-run 通过。
- adapter 单条真实推理通过。
- 1 条 test set quick eval 通过。

### 当前验证

- `python -B -m unittest discover -s tests -v`：22 个测试通过。
- `infer.py` dry-run 通过。
- `evaluate.py` dry-run 通过。
- adapter 单条真实推理通过。
- 1 条 test set quick eval 通过。
- `manual_compare_qwen05b_qlora_3k.jsonl` 已完成 30 条 Base vs Adapter 对比。

### 5.3 固定 prompt 对比

已完成 `manual_prompts_30.jsonl` 上的 Base Model vs QLoRA-3k Adapter 固定 prompt 对比。

输出文件：

```text
results/manual_compare_qwen05b_qlora_3k.jsonl
```

运行命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\manual_prompts_30.jsonl --output_file results\manual_compare_qwen05b_qlora_3k.jsonl --mode both --max_new_tokens 128
```

检查结果：

| 检查项 | 结果 |
|---|---:|
| 样本数 | 30 |
| base 输出缺失数 | 0 |
| adapter 输出缺失数 | 0 |
| base 平均输出长度 | 约 193.5 字符 |
| adapter 平均输出长度 | 约 199.1 字符 |

初步观察：

- Adapter 输出整体更倾向于短段落或编号列表，结构化倾向较明显。
- Base 输出有时更发散，会出现自我身份说明或过度扩展。
- Adapter 仍存在事实错误、概念混淆或生成被 `max_new_tokens=128` 截断的问题。
- 这些观察只是固定 prompt 的人工分析起点，不应直接作为最终效果结论。

结果说明口径：

> 我没有只看训练 loss，而是固定了一批人工 prompt，让 Base Model 和 QLoRA Adapter 在同一输入下生成回答。这样可以观察微调后模型在结构化表达、指令遵循和回答完整性上的变化，同时也能保留失败案例，比如概念混淆或回答被截断。这个环节是为了补足 ROUGE-L 和 eval loss 的局限，让实验结论更可信。

### 5.4 结果整理

已完成第 5 阶段结果整理，将自动评估、固定 prompt 对比和人工分析补充到：

```text
docs/experiment_results.md
README.md
```

人工分析使用完整的 `results/manual_compare_qwen05b_qlora_3k.jsonl`，重点挑选了成功和失败样例，避免只保留好看的输出。

| 样例 | prompt 摘要 | 主要观察 | 结论 |
|---|---|---|---|
| `manual_8244` | 公司内部沟通方案 | Adapter 直接给编号方案，Base 开头有身份说明 | Adapter 结构化更明显 |
| `manual_3223` | 日期抽取 | Adapter 明确列出 `3月1日` 和 `4月30日` | Adapter 更贴近抽取任务 |
| `manual_2884` | 石头剪刀布 Python 程序 | Base 不合理拒答，Adapter 开始生成代码但截断 | Adapter 更愿意执行指令，但完整性不足 |
| `manual_1849` | 太阳系概括 | Base 和 Adapter 都存在明显事实错误 | 微调未解决事实可靠性 |
| `manual_5701` | 设置 iPhone Touch ID | 两者步骤都不准确 | 操作指导仍可能幻觉 |

第 5 阶段最终结论：

- 训练与评估闭环已形成：训练 loss / eval loss、test ROUGE-L、test perplexity、固定 prompt 对比和人工失败案例均已具备。
- Adapter 在部分任务上更倾向于编号列表、短段落和直接回答，说明小规模 SFT 对输出格式和部分指令遵循有帮助。
- Adapter 仍存在事实错误、概念混淆和生成截断，结论必须保守表达为“观察到部分改善”，不能说能力显著提升。

### 自动评估明细

已完成 `test_100.jsonl` 上的 QLoRA-3k 自动评估。

输出文件：

```text
results/predictions_qwen05b_qlora_3k.jsonl
results/metrics_qwen05b_qlora_3k.json
```

运行方式：

1. 先运行完整 100 条生成与 ROUGE-L，跳过 perplexity：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval.yaml --skip_perplexity --max_new_tokens 128 --predictions_file results\predictions_qwen05b_qlora_3k.jsonl --metrics_file results\metrics_qwen05b_qlora_3k_rouge_only.json
```

2. 再复用预测文件，单独补 eval loss / perplexity：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval.yaml --skip_generation --predictions_file results\predictions_qwen05b_qlora_3k.jsonl --metrics_file results\metrics_qwen05b_qlora_3k.json
```

检查结果：

| 指标 | 结果 |
|---|---:|
| test examples | 100 |
| prediction rows | 100 |
| adapter 输出缺失数 | 0 |
| ROUGE-L | 约 0.2539 |
| eval loss | 约 1.8221 |
| perplexity | 约 6.1851 |

解释：

- ROUGE-L 只衡量生成文本与参考答案的字面重合趋势，不能完整代表回答质量。
- eval loss / perplexity 是在 `test_100` 上计算的语言建模指标，和第 4 阶段 `valid_300` 的 eval loss 不完全可直接比较。
- 当前自动评估已经形成闭环，但结论仍需结合固定 prompt 和人工分析。

### 第 5 阶段当前状态

第 5 阶段已完成。下一步进入第 6 阶段：继续完善版本整理材料，包括 README 结果表、可复现实验命令、项目结构说明和推理说明素材。

## 2026-05-30 第 6 阶段前审查与修复

### 审查发现

在进入第 6 阶段版本整理前，对已有工作做了一轮 review，重点检查：

- 6GB 显存约束下的训练配置是否可复现。
- 训练、推理、评估命令是否容易误导复现。
- 大文件、模型权重、训练输出和本地结果是否存在误入 Git 风险。
- README、`docs/`、`AGENTS.md` 的阶段状态是否一致。
- 自动指标和人工分析是否保持保守表达。

### 已修复

| 优先级 | 问题 | 处理 |
|---|---|---|
| P0 | `scripts/train_lora.py` 默认 `train_100`，但配置默认输出到 `outputs/qwen05b_qlora_3k`，容易误写主实验目录 | 将 CLI 默认 `train_split` 调整为 `train_3k`，README 主实验命令显式写出 `--train_split train_3k --output_dir outputs/qwen05b_qlora_3k` |
| P0 | README 推理命令使用训练配置，缺少 `adapter_dir` | 改为 `configs/eval.yaml` 并显式使用 `--mode adapter` |
| P1 | `AGENTS.md` 阶段状态停留在第 4 阶段前 | 更新长期阶段状态：第 4、5 阶段已完成，第 6 阶段为当前下一步 |
| P1 | `results/*.json/jsonl` 被 `.gitignore` 忽略，公开仓库中读者看不到完整本地结果文件 | 新增 `docs/result_samples.md`，摘录指标和少量固定 prompt 样例 |
| P2 | `evaluate.py --skip_generation --limit` 复用预测文件时没有限制预测行数 | 新增 `apply_prediction_limit`，复用预测文件时按 `limit` 截断 |

### 新增验证

- 新增 `tests.test_train_lora.TrainLoraConfigTest.test_cli_default_train_split_matches_main_experiment`，覆盖训练 CLI 默认 split。
- 新增 `tests.test_evaluate_script.EvaluateScriptTest.test_apply_prediction_limit_limits_reused_prediction_rows`，覆盖复用预测文件时的 limit 行为。
- `python -B -m unittest discover -s tests -v`：22 个测试通过。
- `python -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --dry_run`：默认使用 `train_3k`，输出目录为 `outputs\qwen05b_qlora_3k`。

### 仍需注意

- 当前目录还不是 Git 仓库；正式进入版本整理前，需要初始化 Git 后运行 `git status --ignored`，确认 `.gitignore` 生效且没有大文件进入提交范围。
- `eval loss / perplexity` 当前计算口径是完整 chat template 的语言建模 loss，不是只针对 assistant answer 的 loss；README 已保持保守表达，后续如要更严谨可改为 assistant-only loss。

## 2026-05-30 第 6 阶段 Git 初始化与计划

### Git 初始化

已在项目根目录初始化 Git 仓库：

```powershell
git init
```

首次在沙箱内执行 `git init` 时，写 `.git/config` 失败并留下 `.git/config.lock`；清理该失败锁文件后，使用提升权限重新执行 `git init` 成功。

### `.gitignore` 审查

已执行：

```powershell
git status --ignored --short -uall
```

审查结果：

- 原始大数据已被忽略：`data/BelleGroup/`、`data/BelleGroup_sft.jsonl`、`data/BelleGroup_sft_1k.jsonl`、`data/mobvoi_seq_monkey_general_open_corpus.jsonl`、`data/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2`。
- 处理后数据已被忽略：`data/processed/`。
- 本地模型和权重已被忽略：`models/`、`base_model_215M/`、`*.safetensors`、`*.pth`、`*.pt`、`*.bin`。
- 训练输出已被忽略：`outputs/`。
- 本地完整评估结果已被忽略：`results/*.jsonl`、`results/*.json`。
- Python 缓存和测试临时文件已被忽略：`__pycache__/`、`tests/.tmp/`。

额外发现：

- `data/.msc` 是本地下载/缓存痕迹，已加入 `.gitignore`。
- `results/.gitkeep` 仍可纳入版本管理，用于保留结果目录结构；完整 JSON/JSONL 结果继续留在本地。

### 第 6 阶段拆分计划

已新增计划文档：

```text
docs/superpowers/plans/2026-05-30-stage6-github-showcase.md
```

第 6 阶段建议拆成 5 个小阶段：

| 子阶段 | 目标 | 主要产出 |
|---|---|---|
| 6.1 Git ignore audit | 确认大文件、模型、输出和本地结果不会误入 Git | `.gitignore`、`git status --ignored` 记录 |
| 6.2 README polish | 让 GitHub 首页能清楚说明项目目标、运行方式、结果和局限 | `README.md` |
| 6.3 Result summary / demo material | 提供轻量样例和指标摘要 | `docs/result_samples.md`、可选 demo 素材 |
| 6.4 Reproducibility check | 验证轻量命令、环境说明和 dry-run 都可复现 | `docs/setup.md`、测试结果 |
| 6.5 Commit boundary review | 明确哪些文件应该进入第一版提交 | `git status --short`、 staged 文件清单 |

### 本轮执行顺序

优先执行 6.1 和 6.2，随后轻量检查 6.3 和 6.4：

1. 再跑一次 `git status --ignored --short -uall`，确认 `data/.msc` 已被忽略。
2. 整理 README 的公开仓库首页结构，减少重复段落，把“快速开始”和“实验结果”变成第 6 阶段版本。

### 6.1 / 6.2 执行结果

已完成：

- `6.1 Git ignore audit`：再次运行 `git status --ignored --short -uall`，确认 `models/`、`outputs/`、`data/processed/`、原始大数据、完整 `results/*.json/jsonl`、缓存和测试临时文件均处于 ignored 状态。
- `6.2 README polish`：重写 README 为公开仓库版本，首屏直接说明项目定位、硬件约束、已完成阶段、实验配置、训练/评估结果、快速开始、项目结构和大文件策略。
- README 中主实验、推理、评估命令已统一使用本机 `pytorch_env` 示例，并显式指定 `train_3k`、`outputs/qwen05b_qlora_3k` 和 `configs/eval.yaml`。
- README 已移除容易过度承诺的可视化表述，保留命令行推理和结果样例。

本轮验证结果：

- `python -B -m unittest discover -s tests -v`：22 个测试全部通过。
- `python -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --dry_run`：确认默认主实验读取 `train_3k`、`valid_300`，输出到 `outputs/qwen05b_qlora_3k`。
- `python -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\manual_prompts_30.jsonl --limit 2 --mode both --dry_run`：推理配置 dry-run 通过。
- `python -B scripts\evaluate.py --config configs\eval.yaml --limit 2 --dry_run`：评估配置 dry-run 通过。
- `git status --ignored --short -uall`：确认大模型、训练输出、处理后数据、完整结果 JSON/JSONL、缓存和测试临时文件仍被忽略。

当前第 6 阶段剩余建议：

1. `6.3 Result summary / demo material` 已轻量检查：`docs/result_samples.md` 与本地 `results/metrics_qwen05b_qlora_3k.json` 指标一致，包含结构化输出、抽取任务和失败案例 3 类样例；本轮未新增额外 demo 素材，避免提交大结果文件。
2. `6.4 Reproducibility check` 已完成轻量检查：单元测试、训练 dry-run、推理 dry-run、评估 dry-run 均通过；下一轮可视情况补跑 `pytorch_env` 的 `--check_env` 重依赖检查。
3. 执行 `6.5 Commit boundary review`：决定是否把 `tiny_llm_legacy/` 和 `tokenizer_k/` 纳入首版 GitHub 提交。

## 2026-05-31 第 6 阶段提交边界审查

### 已完成

- 补跑 `pytorch_env` 重依赖检查：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --check_env
```

检查结果：

| 项目 | 结果 |
|---|---|
| 环境检查 | `ENV_CHECK_OK` |
| PyTorch | `2.10.0+cu126` |
| CUDA | 可用 |
| GPU | `NVIDIA GeForce RTX 3060 Laptop GPU` |

- 执行 `6.5 Commit boundary review`，确认首版提交应聚焦 Qwen2.5-0.5B QLoRA 中文指令微调闭环。
- `tiny_llm_legacy/` 当前只作为旧 Tiny LLM 学习代码和历史参考，不被 README 主线命令、训练脚本、推理脚本或评估脚本依赖；首版 GitHub 建议不提交，避免分散项目定位。
- `tokenizer_k/` 是旧自训 tokenizer 产物，和当前 `models/qwen2.5-0.5b-instruct/` 的 Qwen tokenizer 无关；首版 GitHub 建议不提交。
- `git status --ignored --short -uall` 显示模型权重、训练输出、处理后数据、完整 `results/*.json/jsonl`、缓存和测试临时文件仍处于 ignored 状态。

### 首版提交建议

建议第一版只提交：

```text
.gitignore
AGENTS.md
README.md
requirements.txt
configs/
docs/
scripts/
src/
tests/
assets/.gitkeep
data/samples/.gitkeep
results/.gitkeep
```

建议暂不提交：

```text
tiny_llm_legacy/
tokenizer_k/
```

继续禁止提交：

```text
models/
outputs/
data/processed/
data/BelleGroup/
data/BelleGroup_sft*.jsonl
results/*.json
results/*.jsonl
base_model_215M/
*.safetensors
*.pth
*.pt
*.bin
```

### 当前状态

第 6 阶段版本整理材料和提交边界已基本收尾。下一步可以按建议清单进行有意 staging，并在 staged 状态下再次检查 `git status --short`，确认没有模型、数据和完整实验产物进入提交。

## 2026-05-31 第 6 阶段完成与第 7.1 启动

### 第 6 阶段完成状态

- 已完成首版 GitHub 提交：

```text
83ee7e5 Initial QLoRA instruction tuning project
```

- 已绑定远程仓库并推送到 `main`：

```text
https://github.com/KOIIIII0525/-LoRA-QLoRA-.git
```

- `git status -sb` 显示本地 `main` 已跟踪 `origin/main`，无未提交主线文件。
- 首版提交不包含模型权重、原始大数据、`data/processed/`、`outputs/` 或完整 `results/*.json/jsonl`。
- `tiny_llm_legacy/` 和 `tokenizer_k/` 已作为本地旧项目资料保留并忽略，不进入首版 GitHub。

### 第 7.1 版本微调范围

第 7 阶段先从轻量版本微调开始，不启动重训练，不引入新依赖。

本轮 7.1 只做公开仓库状态更新：

- 更新 `README.md` 阶段表，标记第 6 阶段已完成，第 7 阶段作为可选加分项待推进。
- 更新 `docs/progress.md` 阶段总览，记录首版 GitHub 推送完成。
- 不新增私人用途文档，避免把与项目复现无关的内容放入公开仓库。

后续第 7 阶段建议继续按低风险优先级推进：

1. 检查 GitHub 页面 README 渲染效果和仓库描述。
2. 视需要做本地 Demo 增强，但不提交模型或 adapter 权重。
3. 如果时间充足，再考虑 5k 数据或 LoRA rank 消融等重实验。

## 2026-05-31 第 7 阶段轻量收束

### 本轮判断

当前项目已经具备本地命令行推理能力，不再把 Gradio Demo 作为第一版结尾的必要任务。项目主线仍然聚焦：

```text
数据处理 -> QLoRA 训练 -> 推理对比 -> 自动评估 -> 人工分析 -> 版本整理
```

已验证真实 adapter 单条推理命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --prompt "请用三句话说明什么是参数高效微调。" --mode adapter --max_new_tokens 96
```

结果：命令成功加载本地 `models/qwen2.5-0.5b-instruct/` 和 `outputs/qwen05b_qlora_3k/` adapter，并返回中文回答。该命令可作为轻量本地推理入口；首次运行需要加载模型权重，耗时明显长于 dry-run。

### Demo 取舍

- 不新增 Gradio Demo，避免把项目重心从“中文指令微调实验闭环”转向 Web UI。
- 项目说明优先使用 README、结果表、固定 prompt 对比、失败案例和终端单条推理命令。
- 如后续确实需要更强交互能力，可在不提交模型权重的前提下补充 Gradio Demo。

### rank4 消融状态

- 已提交 `configs/lora_qwen_0.5b_rank4.yaml` 和 `configs/eval_rank4.yaml`。
- 本地 RTX 3060 6GB 上 rank4 训练速度异常慢，暂不把消融结果写入第一版结论。
- 后续如租用服务器，可继续完成 rank4 训练、自动评估和 manual prompts 对比，再与当前 rank8 主实验做保守对比。

### 当前收尾结论

第一版项目已经达到最初预期：在消费级显卡约束下完成轻量中文指令微调、评估和本地推理闭环。后续扩展可作为补充实验，不影响当前版本的复现和结果说明。

## 2026-06-01 rank4-1k 消融实验

### 实验目标

本轮将 LoRA rank 消融从原计划的 `rank4-3k` 先调整为更轻量的 `rank4-1k`，用于快速验证 rank4 配置、输出路径和评估闭环是否稳定。

### 已完成

- 清理旧的 `outputs/qwen05b_qlora_3k_r4/` 失败残留，只保留成功的 `smoke`、`qwen05b_qlora_1k` 和 `qwen05b_qlora_3k` 输出目录。
- 完成 `rank=4`、`train_1k` 训练，输出目录为 `outputs/qwen05b_qlora_1k_r4/`。
- 新增 `configs/eval_rank4_1k.yaml`，将 rank4-1k 评估结果隔离到独立文件名。
- 完成 `test_100` 自动评估：
  - `results/predictions_qwen05b_qlora_1k_r4.jsonl`
  - `results/metrics_qwen05b_qlora_1k_r4_rouge_only.json`
  - `results/metrics_qwen05b_qlora_1k_r4.json`
- 完成 `manual_prompts_30` 固定 prompt 对比：
  - `results/manual_compare_qwen05b_qlora_1k_r4.jsonl`
- 新增 `docs/rank4_1k_ablation.md`，集中记录 rank4-1k 的配置、训练结果、自动指标、manual prompt 完整性和对比边界。

### 关键结果

| 实验 | step | valid eval loss | test ROUGE-L | test eval loss | test ppl |
|---|---:|---:|---:|---:|---:|
| QLoRA-1k-rank4 | 125 / 125 | 约 1.7739 | 0.2195 | 1.8499 | 6.3592 |
| QLoRA-1k-rank8 | 125 / 125 | 约 1.7662 | 暂未补跑 | 暂未补跑 | 暂未补跑 |

### 当前结论

- 在同为 `train_1k` 的训练指标上，rank8 的 valid eval loss 略低于 rank4，但差距较小。
- rank4-1k 的自动评估和固定 prompt 对比已完成，文件命名独立，没有覆盖 rank8 主线结果。
- 严格的 rank4-vs-rank8 自动指标对比还需要补跑 `outputs/qwen05b_qlora_1k/` 的 `test_100` 评估；如果优先推进主消融，则下一步更建议直接跑 `rank4-3k`，因为 rank8-3k 已经有完整自动评估和人工分析结果。

### 下一步

准备 `rank=4`、`train_3k` 主消融实验，输出目录继续使用 `outputs/qwen05b_qlora_3k_r4/`，评估配置使用已存在的 `configs/eval_rank4.yaml`。

## 2026-06-02 第 8 阶段：DPO 偏好对齐启动

### 项目定位更新

项目从单纯的 QLoRA 中文指令微调项目，升级为：

```text
基于 Qwen2.5-0.5B 的中文指令微调与偏好对齐实验项目
```

新主线：

```text
数据处理 -> QLoRA SFT -> Base/SFT 评估 -> 偏好数据构造 -> DPO 对齐 -> Base/SFT/DPO 对比
```

当前仍以 `Qwen2.5-0.5B-Instruct`、RTX 3060 Laptop 6GB、低显存可复现为边界，不把 PPO 或完整人类 RLHF 作为第一版主线。DPO 阶段表述为“小规模离线偏好对齐 / RLAIF 风格实验”。

### 本轮新增

| 文件 | 作用 |
|---|---|
| `configs/dpo_qwen_0.5b.yaml` | DPO 偏好数据、后续训练参数和输出路径配置 |
| `src/preference_utils.py` | 抽取 prompt/reference、构造 chosen/rejected、校验偏好样本和确定性划分 |
| `scripts/prepare_preference_data.py` | 生成偏好 prompt 池、将 adapter 预测结果转换为 DPO 数据 |
| `tests/test_preference_utils.py` | 偏好工具单元测试 |
| `tests/test_prepare_preference_data.py` | 偏好数据脚本单元测试 |
| `docs/dpo_alignment_plan.md` | DPO 阶段外部记忆和操作文档 |

### 偏好数据策略

第一版偏好数据：

```text
chosen = BELLE 原始参考答案
rejected = QLoRA-3k SFT adapter 生成答案
```

计划产出：

| 文件 | 用途 |
|---|---|
| `data/processed/preference_train_50.jsonl` | DPO smoke test |
| `data/processed/preference_train_300.jsonl` | DPO 主实验 |
| `data/processed/preference_valid_100.jsonl` | DPO 验证 |
| `data/processed/preference_test_100.jsonl` | DPO 偏好评估 |

划分关系：

- `preference_train_50` 是 `preference_train_300` 的子集。
- `preference_valid_100`、`preference_test_100` 与 `preference_train_300` 不重叠。
- 偏好 prompt 池不复用已有 `train_100 / train_1k / train_3k / valid_300 / test_100 / manual_prompts_30`。

### 本轮已生成的本地产物

| 文件 | 状态 |
|---|---|
| `data/processed/preference_prompts_pool.jsonl` | 已重新生成 500 行，与旧 SFT/评估 split 重叠为 0 |
| `results/preference_rejected_qwen05b_qlora_3k_quick.jsonl` | 已完成 1 条 adapter rejected 真实生成冒烟 |
| `results/preference_rejected_chunk_000.jsonl` | 已完成 50 条 adapter rejected 分批生成 |
| `data/processed/preference_train_50.jsonl` | 已完成 50 条 DPO smoke 数据 |

质量检查：

| 检查项 | 结果 |
|---|---:|
| preference prompt pool 行数 | 500 |
| 与 `train_100` prompt 重叠 | 0 |
| 与 `train_1k` prompt 重叠 | 0 |
| 与 `train_3k` prompt 重叠 | 0 |
| 与 `valid_300` prompt 重叠 | 0 |
| 与 `test_100` prompt 重叠 | 0 |
| 与 `manual_prompts_30` prompt 重叠 | 0 |

### 本轮验证

新增测试先按 TDD 红灯确认，再补实现。当前验证结果：

```powershell
python -B -m unittest tests.test_preference_utils -v
python -B -m unittest tests.test_prepare_preference_data -v
```

两组新增测试均通过。

偏好数据 dry-run：

```powershell
python -B scripts\prepare_preference_data.py --config configs\dpo_qwen_0.5b.yaml --stage prompts --dry_run
python -B scripts\prepare_preference_data.py --config configs\dpo_qwen_0.5b.yaml --stage pairs --dry_run
```

真实 prompt 池生成：

```powershell
python -B scripts\prepare_preference_data.py --config configs\dpo_qwen_0.5b.yaml --stage prompts
```

1 条 adapter rejected 冒烟：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\preference_prompts_pool.jsonl --output_file results\preference_rejected_qwen05b_qlora_3k_quick.jsonl --mode adapter --limit 1 --max_new_tokens 64
```

结果：命令成功生成 1 行 `prompt/reference/adapter_response`，说明现有 `infer.py` 可以复用为 DPO rejected response 生成入口。

### 数据隔离审核与长任务处理

用户要求审核 DPO 训练集是否单独。初次审核发现旧 prompt 池和 `train_3k`、`valid_300` 存在重叠，因此已将所有旧 SFT/评估 split 加入 `configs/dpo_qwen_0.5b.yaml` 的 `exclude_prompt_files`，并新增测试覆盖该配置要求。

重新生成后，`preference_prompts_pool.jsonl` 与 `train_100 / train_1k / train_3k / valid_300 / test_100 / manual_prompts_30` 的 prompt 重叠均为 0。结论：DPO 偏好数据当前是单独 held-out 数据。

一次性生成 500 条 rejected responses 时，进程正常使用 GPU，但由于 `infer.py` 原本会在全部生成后一次性写文件，中途不可观察，运行时间过长。已停止该长任务，并为 `infer.py` 新增 `--offset`，支持分批生成。

第 1 批 50 条已生成：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\preference_prompts_pool.jsonl --output_file results\preference_rejected_chunk_000.jsonl --mode adapter --offset 0 --limit 50 --max_new_tokens 128
```

随后已生成：

```powershell
python -B scripts\prepare_preference_data.py --config configs\dpo_qwen_0.5b.yaml --stage pairs --only_split preference_train_50 --predictions_file results\preference_rejected_chunk_000.jsonl --output_file data\processed\preference_train_50.jsonl
```

`preference_train_50` 审核结果：

| 检查项 | 结果 |
|---|---:|
| rows | 50 |
| empty chosen | 0 |
| empty rejected | 0 |
| chosen/rejected 完全相同 | 0 |
| 与 `train_3k` prompt 重叠 | 0 |
| 与 `valid_300` prompt 重叠 | 0 |
| 与 `test_100` prompt 重叠 | 0 |
| 与 `manual_prompts_30` prompt 重叠 | 0 |

### 下一步

1. 新增 `scripts/train_dpo.py`，先跑 `preference_train_50` smoke test。
2. smoke test 稳定后，再继续分批生成剩余 rejected chunks。
3. 凑齐 500 条后，再生成 `preference_train_300 / preference_valid_100 / preference_test_100`。

### 2026-06-02 DPO smoke test 完成

本轮新增：

| 文件 | 作用 |
|---|---|
| `src/dpo_utils.py` | DPO 配置解析、路径解析、schema 校验、dry-run 报告和 DPOConfig 参数兼容 |
| `scripts/train_dpo.py` | DPO 训练入口，支持 `--dry_run`、`--check_env` 和真实训练 |
| `tests/test_dpo_utils.py` | DPO 工具测试 |
| `tests/test_train_dpo.py` | DPO 训练脚本测试 |
| `configs/eval_dpo_50.yaml` | DPO smoke adapter 推理/评估配置 |

依赖：

- `pytorch_env` 初次检查缺少 `trl`。
- 已安装 `trl==1.5.1`。
- `D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_50 --check_env`：通过。

DPO dry-run：

```powershell
python -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_50 --dry_run
```

确认：

| 项目 | 值 |
|---|---|
| train file | `data/processed/preference_train_50.jsonl` |
| train rows | 50 |
| eval rows | 0 |
| output dir | `outputs/qwen05b_dpo_50` |
| beta | 0.1 |
| max length | 512 |
| batch size | 1 |
| grad accumulation | 8 |
| learning rate | 5e-5 |

训练问题与修复：

- 第一次真实 DPO 训练在 `fp16=true` 下报 `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`。
- 检查可训练 LoRA 参数 dtype 为 `float32`，判断问题来自当前 TRL/Transformers/Accelerate/PyTorch 组合下 AMP GradScaler 与 bfloat16 梯度路径不兼容。
- 已将 DPO 阶段配置 `fp16` 改为 `false`，仍保持 4-bit QLoRA 加载。

真实 DPO smoke test 命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_50 --output_dir outputs\qwen05b_dpo_50
```

结果：

| 指标 | 结果 |
|---|---:|
| optimizer steps | 7 |
| train runtime | 约 38 秒 |
| train loss | 约 0.6355 |
| rewards/accuracies | 约 0.84 |
| rewards/margins | 约 0.1384 |

产物：

| 文件 | 状态 |
|---|---|
| `outputs/qwen05b_dpo_50/adapter_model.safetensors` | 已生成 |
| `outputs/qwen05b_dpo_50/checkpoint-7/` | 已生成 |

DPO adapter 推理验证：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval_dpo_50.yaml --prompt "请用三句话说明什么是参数高效微调。" --mode adapter --max_new_tokens 64
```

结果：命令成功加载 `outputs/qwen05b_dpo_50/` 并返回中文回答，说明 DPO smoke adapter 可用于后续推理/评估。

下一步：

1. 继续用 `--offset/--limit` 分批生成剩余 rejected chunks。
2. 凑齐完整 500 条后，生成 `preference_train_300 / preference_valid_100 / preference_test_100`。
3. 跑 `preference_train_300` DPO 主实验。
4. 做 Base / SFT / DPO 三模型自动指标和固定 prompt 人工对比。

### 2026-06-02 DPO-300 主实验完成

分批生成 rejected responses 已完成：

| 文件范围 | 状态 |
|---|---|
| `results/preference_rejected_chunk_000.jsonl` - `chunk_009.jsonl` | 均为 50 行 |
| `results/preference_rejected_qwen05b_qlora_3k.jsonl` | 已合并 500 行 |

已生成完整 preference split：

| 文件 | 行数 | 质量检查 |
|---|---:|---|
| `data/processed/preference_train_50.jsonl` | 50 | empty chosen/rejected 为 0，chosen/rejected 相同为 0 |
| `data/processed/preference_train_300.jsonl` | 300 | empty chosen/rejected 为 0，chosen/rejected 相同为 0 |
| `data/processed/preference_valid_100.jsonl` | 100 | empty chosen/rejected 为 0，chosen/rejected 相同为 0 |
| `data/processed/preference_test_100.jsonl` | 100 | empty chosen/rejected 为 0，chosen/rejected 相同为 0 |

划分检查：

| 检查项 | 结果 |
|---|---:|
| `preference_train_50` 是 `preference_train_300` 子集 | True |
| `train_300` 与 `valid_100` prompt 重叠 | 0 |
| `train_300` 与 `test_100` prompt 重叠 | 0 |
| `valid_100` 与 `test_100` prompt 重叠 | 0 |

DPO-300 dry-run：

```powershell
python -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_300 --dry_run
```

确认读取：

| 项目 | 值 |
|---|---|
| train file | `data/processed/preference_train_300.jsonl` |
| train rows | 300 |
| eval file | `data/processed/preference_valid_100.jsonl` |
| eval rows | 100 |
| output dir | `outputs/qwen05b_dpo_300` |

真实训练命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_dpo.py --config configs\dpo_qwen_0.5b.yaml --train_split preference_train_300 --output_dir outputs\qwen05b_dpo_300
```

结果：

| 指标 | 结果 |
|---|---:|
| optimizer steps | 38 |
| train runtime | 约 13 分 19 秒 |
| train loss | 约 0.419 |
| eval loss | 约 0.3049 |
| eval rewards/accuracies | 约 0.93 |
| eval rewards/margins | 约 1.2685 |

产物：

| 文件 | 状态 |
|---|---|
| `outputs/qwen05b_dpo_300/adapter_model.safetensors` | 已生成 |
| `outputs/qwen05b_dpo_300/checkpoint-38/` | 已生成 |
| `configs/eval_dpo_300.yaml` | 已新增 |

DPO-300 adapter 推理验证：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval_dpo_300.yaml --prompt "请用三句话说明什么是参数高效微调。" --mode adapter --max_new_tokens 64
```

结果：命令成功加载 `outputs/qwen05b_dpo_300/` 并返回中文回答。

下一步进入三模型评估阶段：

1. DPO-300 在 `test_100` 上跑自动评估。
2. 在 `manual_prompts_30` 上生成 SFT vs DPO 固定 prompt 对比。
3. 增加 preference accuracy，比较 SFT/DPO 对 `preference_test_100` 中 chosen/rejected 的偏好。
4. 整理 `docs/dpo_alignment_results.md` 和 README。
