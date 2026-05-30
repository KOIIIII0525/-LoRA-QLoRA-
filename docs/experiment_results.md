# 实验结果记录

本文档用于集中记录第一版 QLoRA 主线实验结果，方便后续写 README、结果表和面试讲解。

## 实验目标

第 4 阶段的目标是在 `train_100` smoke test 成功的基础上，逐步扩大训练数据规模，验证项目是否能在 RTX 3060 Laptop 6GB 约束下完成可复现的中文指令微调主实验。

核心问题：

```text
当训练数据从 100 条扩大到 1k、3k 时，训练流程是否稳定，验证集 eval loss 是否有可观察变化？
```

## 统一实验配置

| 配置项 | 值 |
|---|---|
| base model | `models/qwen2.5-0.5b-instruct` |
| tuning method | QLoRA / LoRA adapter |
| quantization | 4-bit |
| eval split | `data/processed/valid_300.jsonl` |
| eval rows | 300 |
| epoch | 1 |
| max seq len | 512 |
| per-device train batch size | 1 |
| gradient accumulation | 8 |
| learning rate | 2e-4 |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| gradient checkpointing | enabled |
| precision | fp16 |

说明：`train_100` 只作为 smoke test，用于证明流程可跑通；`train_1k` 和 `train_3k` 才用于观察数据规模扩大后的训练指标变化。

## 阶段 4 结果表

| 实验 | 训练数据 | 输出目录 | step | 最终 eval loss | logged train loss 均值 | 最后一次 logged train loss | 产物状态 |
|---|---:|---|---:|---:|---:|---:|---|
| QLoRA-smoke | 100 | `outputs/smoke/` | 13 / 13 | 1.9051 | 2.0235 | 2.0235 | 已生成 adapter |
| QLoRA-1k | 1000 | `outputs/qwen05b_qlora_1k/` | 125 / 125 | 1.7662 | 1.8355 | 1.7923 | 已生成 adapter |
| QLoRA-3k | 3000 | `outputs/qwen05b_qlora_3k/` | 375 / 375 | 1.7448 | 1.8104 | 1.7785 | 已生成 adapter |

## 3k 训练过程指标

| checkpoint | epoch | eval loss | 说明 |
|---|---:|---:|---|
| step 100 | 0.2667 | 1.7682 | 第一次验证，与 1k 最终 eval loss 接近 |
| step 200 | 0.5333 | 1.7534 | 验证集 loss 继续下降 |
| step 300 | 0.8000 | 1.7474 | 接近最终结果 |
| step 375 | 1.0000 | 1.7448 | 3k 主实验最终验证结果 |

训练日志显示 `train_3k` 总耗时约 6 小时 28 分钟，`train_samples_per_second` 约 0.129，`train_steps_per_second` 约 0.016。训练期间 6GB 显存基本打满，说明当前配置能跑通，但效率并不高。

## 训练阶段结论

1. `train_100` 证明训练链路可用，包括模型加载、数据格式、QLoRA、Trainer、checkpoint 和 adapter 保存。
2. `train_1k` 完成完整 1 epoch，最终 `eval_loss` 从 smoke test 的约 1.9051 降到约 1.7662，说明更大训练集在固定验证集上带来明显 loss 下降。
3. `train_3k` 完成完整 1 epoch，最终 `eval_loss` 约 1.7448，相比 1k 的约 1.7662 继续小幅下降。
4. 3k 相比 1k 的提升幅度变小，说明小规模 SFT 的收益可能存在边际递减，后续需要结合生成质量和人工分析判断是否值得继续扩大数据。
5. 训练阶段结论只能说明验证集语言建模 loss 下降，不能直接等同于“指令遵循能力显著提升”。因此第 5 阶段已补充固定 prompt 对比、ROUGE-L 和人工分析。

## 面试讲解口径

可以这样讲：

> 我把主实验拆成 100、1k、3k 三个规模。100 条只做 smoke test，确保模型、数据、QLoRA 和保存流程跑通；1k 用来验证小规模 SFT 是否稳定，并拿到第一组有效 eval loss；3k 作为第一版主实验结果，用来观察扩大数据后的变化。结果上，eval loss 从 smoke 的约 1.905 降到 1k 的约 1.766，再到 3k 的约 1.745，说明在固定验证集上有持续但逐渐变小的改善。考虑到自动 loss 不能完全代表问答质量，所以后续还需要固定 prompt、ROUGE-L 和人工维度评估。

## 关键概念

- QLoRA：用 4-bit 量化加载基座模型，只训练 LoRA adapter，从而降低显存占用。
- LoRA adapter：微调得到的增量参数，推理时与 base model 组合使用，不需要保存完整模型副本。
- eval loss：模型在未参与训练的验证集上的语言建模损失，用于观察泛化趋势。
- checkpoint：训练中间保存点，用于恢复训练、检查过程指标或选择不同 step 的 adapter。
- logged train loss：训练过程中按 `logging_steps` 记录的局部 loss，用于观察训练是否异常震荡。
- 边际收益递减：从 100 到 1k 的 loss 改善更明显，从 1k 到 3k 改善变小，后续扩大数据时需要考虑时间成本和效果收益。

## 评估闭环状态

第 5 阶段已补齐第一版评估闭环：

1. `scripts/infer.py` 已支持 base / adapter / both 推理，并生成固定 prompt 对比文件。
2. `scripts/evaluate.py` 已支持 test set 预测、ROUGE-L、eval loss 和 perplexity。
3. `manual_prompts_30.jsonl` 已生成 Base Model vs QLoRA-3k 对比结果。
4. 已完成自动指标、固定 prompt 样例和人工分析整理。

## 第 5 阶段进展

### 5.1 / 5.2 脚本状态

已新增：

| 文件 | 作用 |
|---|---|
| `src/eval_utils.py` | 评估与推理的纯工具函数，负责样本规范化、结果记录和 loss 摘要 |
| `scripts/infer.py` | 支持 base / adapter / both 的单条或批量推理 |
| `scripts/evaluate.py` | 支持 test set 预测、ROUGE-L 和可选 perplexity |
| `tests/test_eval_utils.py` | 纯工具函数测试 |
| `tests/test_infer_script.py` | 推理脚本参数解析测试 |
| `tests/test_evaluate_script.py` | 评估脚本参数解析测试 |

验证结果：

| 验证项 | 结果 |
|---|---|
| 单元测试 | `python -B -m unittest discover -s tests -v`：22 个测试通过 |
| `infer.py` dry-run | 通过，确认 manual prompt 输入和输出路径 |
| `evaluate.py` dry-run | 通过，确认 test set 输入和指标输出路径 |
| adapter 单条真实推理 | 通过 |
| 1 条 quick eval | 通过，ROUGE-L 约 0.3158，perplexity 暂跳过 |

### 初步失败案例

adapter 单条推理 prompt：

```text
请用三句话解释什么是 LoRA。
```

观察：

- 模型将 LoRA 解释成无线通信中的 LoRa，而不是大模型微调中的 Low-Rank Adaptation。
- 这说明当前训练数据和训练 loss 的改善并不能保证模型掌握特定算法概念。
- 该案例适合放入人工分析中的“概念混淆 / 领域幻觉”类别。

### 第 5 阶段结果整理状态

已完成。后续进入第 6 阶段，重点是 README 展示、结果表、项目截图或 Demo 素材整理。

### 5.3 固定 prompt 对比结果

已生成：

```text
results/manual_compare_qwen05b_qlora_3k.jsonl
```

运行配置：

| 项目 | 值 |
|---|---|
| prompt file | `data/processed/manual_prompts_30.jsonl` |
| mode | `both`，顺序运行 Base Model 和 QLoRA-3k Adapter |
| max new tokens | 128 |
| output file | `results/manual_compare_qwen05b_qlora_3k.jsonl` |

完整性检查：

| 检查项 | 结果 |
|---|---:|
| 样本数 | 30 |
| base response 缺失数 | 0 |
| adapter response 缺失数 | 0 |
| base 平均输出长度 | 约 193.5 字符 |
| adapter 平均输出长度 | 约 199.1 字符 |

初步人工观察：

1. Adapter 输出更常见编号列表和短段落，结构化倾向比 Base 更明显。
2. Base 输出有时更发散，部分回答会先解释自己身份或扩展到不必要背景。
3. Adapter 并非总是更好，仍存在事实错误、概念混淆和回答被长度上限截断的问题。
4. 固定 prompt 对比可以提供比单一 loss 更直观的展示材料，但仍需要结合 ROUGE-L 和人工评分表做最终结论。

可用于面试的说明：

> 固定 prompt 对比不是为了挑选好看的案例，而是为了用同一批输入观察 Base Model 和 Adapter 的差异。我保留了完整 30 条输出，并记录失败样例，避免只展示成功案例。初步看，Adapter 在结构化输出上更稳定一些，但仍会出现事实错误或概念混淆，因此后续还要结合自动指标和人工维度分析。

### 5.4 结果整理：自动评估与人工分析

已生成：

```text
results/predictions_qwen05b_qlora_3k.jsonl
results/metrics_qwen05b_qlora_3k.json
```

评估数据：

| 项目 | 值 |
|---|---|
| test file | `data/processed/test_100.jsonl` |
| examples | 100 |
| model | `outputs/qwen05b_qlora_3k/` adapter |
| max new tokens | 128 |

自动指标：

| 指标 | 值 |
|---|---:|
| ROUGE-L | 0.2539 |
| eval loss | 1.8221 |
| perplexity | 6.1851 |

完整性检查：

| 检查项 | 结果 |
|---|---:|
| prediction rows | 100 |
| adapter response 缺失数 | 0 |

指标解释：

- ROUGE-L 反映生成答案与参考答案在最长公共子序列上的重合度，适合做粗粒度自动对比，但对同义表达、结构差异和事实正确性不敏感。
- eval loss / perplexity 衡量模型对测试答案文本的语言建模难度，越低通常越好，但不等同于生成回答一定更符合用户需求。
- 由于 `test_100` 与 `valid_300` 是不同集合，这里的 eval loss 不应直接和训练阶段 `valid_300` 的 eval loss 做严格数值对比，只能作为测试集上的补充指标。

可用于面试的说明：

> 在自动评估环节，我没有只依赖生成样例，而是在固定 test set 上保存完整预测，并计算 ROUGE-L、eval loss 和 perplexity。ROUGE-L 约 0.254，说明模型输出和参考答案有一定字面重合，但这个指标不能判断事实正确性和指令遵循质量。因此我把它作为辅助指标，最终仍结合固定 prompt 对比和人工分析得出保守结论。

### 固定 prompt 人工分析

人工分析基于完整的 `results/manual_compare_qwen05b_qlora_3k.jsonl`，没有只挑选成功案例。当前观察重点放在指令遵循、结构化输出、回答完整性、中文表达和错误类型上。

| 样例 | prompt 摘要 | Base 观察 | Adapter 观察 | 结论 |
|---|---|---|---|---|
| `manual_8244` | 分析公司内部沟通问题并提出方案 | 开头出现身份说明，回答较泛 | 直接给出编号方案，结构更清楚，但被长度上限截断 | Adapter 的结构化倾向更明显 |
| `manual_3223` | 从文本中提取日期 | 给出时间范围，但夹杂额外说明 | 明确列出 `3月1日`、`4月30日` | Adapter 更贴近抽取任务 |
| `manual_2884` | 写石头剪刀布 Python 程序 | 不合理拒答 | 开始生成代码，但因 `max_new_tokens=128` 截断 | Adapter 更愿意执行代码生成指令，但完整性不足 |
| `manual_1849` | 简洁概括太阳系 | 存在明显事实错误 | 同样存在事实错误，如把太阳描述成气体行星/最大行星 | 微调没有解决事实可靠性问题 |
| `manual_5701` | 设置 iPhone Touch ID | 步骤不准确 | 步骤仍不准确，出现不存在的应用入口 | 生活常识类操作指导仍可能幻觉 |

### 第一版评估结论

1. 训练指标显示 `train_100 -> train_1k -> train_3k` 的验证集 loss 逐步下降，说明小规模 QLoRA SFT 流程可运行且训练稳定。
2. 自动评估在 `test_100` 上得到 ROUGE-L 约 `0.2539`、eval loss 约 `1.8221`、perplexity 约 `6.1851`，可作为固定测试集上的辅助证据。
3. 固定 prompt 对比显示 Adapter 更容易输出编号列表、短段落和任务导向回答，部分场景比 Base 更少自我身份说明或无关扩展。
4. Adapter 仍存在事实错误、概念混淆、操作步骤幻觉和生成截断，不能把 loss 下降包装成真实能力显著提升。
5. 第一版项目最稳妥的结论是：在 6GB 消费级显卡约束下，已完成轻量中文指令微调和评估闭环，并观察到小数据 SFT 对回答格式和部分指令遵循有帮助，但效果有限，需要用失败案例说明边界。

### README 展示口径

可以在 README 或面试中这样表达：

> 我把评估拆成训练指标、自动指标和固定 prompt 人工分析三层。训练上，3k QLoRA 的验证集 eval loss 相比 1k 小幅下降；自动评估上，test_100 的 ROUGE-L 约 0.254，perplexity 约 6.185；人工对比上，Adapter 在结构化输出和部分任务遵循上更稳定，但仍有事实错误、概念混淆和截断问题。因此我只把结论表述为“小规模 SFT 对部分指令格式和回答风格有改善”，而不是夸大成模型能力显著提升。
