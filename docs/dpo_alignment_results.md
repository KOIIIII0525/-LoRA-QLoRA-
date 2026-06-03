# DPO 偏好对齐结果记录

本文档记录 Qwen2.5-0.5B 在 SFT-3k adapter 基础上进行小规模 DPO 偏好对齐后的评估结果。当前实验应保守表述为“小规模离线偏好对齐 / RLAIF 风格实验”，不要描述为完整 RLHF。

## 实验设置

| 项目 | 设置 |
|---|---|
| base model | `models/qwen2.5-0.5b-instruct` |
| SFT adapter | `outputs/qwen05b_qlora_3k/` |
| DPO adapter | `outputs/qwen05b_dpo_300/` |
| DPO train split | `data/processed/preference_train_300.jsonl` |
| DPO valid split | `data/processed/preference_valid_100.jsonl` |
| DPO preference test split | `data/processed/preference_test_100.jsonl` |
| preference data | `chosen = BELLE reference`，`rejected = SFT adapter response` |
| scoring method | mean answer token logprob |

## DPO 训练结果

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

## `test_100` 自动评估

| 模型 | ROUGE-L | eval loss | perplexity |
|---|---:|---:|---:|
| SFT-3k adapter | 0.2539 | 1.8221 | 6.1851 |
| DPO-300 adapter | 0.2436 | 1.8466 | 6.3381 |

观察：

- DPO-300 在原 `test_100` 上的 ROUGE-L、eval loss、perplexity 都没有超过 SFT-3k。
- 这说明 DPO 对齐并不等价于通用自动指标提升；当前不能说 DPO 全面提升了模型能力。
- `test_100` 仍是 SFT 阶段的参考答案评估集，而 DPO 的直接目标是偏好排序，因此需要结合 preference accuracy 解读。

## 固定 prompt 对比

已生成：

```text
results/manual_compare_sft_dpo_qwen05b.jsonl
```

完整性检查：

| 检查项 | 结果 |
|---|---:|
| prompt 数量 | 30 |
| SFT response 缺失 | 0 |
| DPO response 缺失 | 0 |
| SFT 平均输出长度 | 约 306.4 字符 |
| DPO 平均输出长度 | 约 192.2 字符 |
| DPO 比 SFT 更短的样例 | 24 / 30 |

人工分析基于完整 30 条 fixed prompt 对比，不只挑选成功案例。

| 类型 | 样例 | SFT 观察 | DPO 观察 | 结论 |
|---|---|---|---|---|
| DPO 形式改善 | `manual_1558` 秋天诗歌 | 更像散文说明，包含“希望满足需求”等助手套话 | 直接生成分行诗歌，意象更集中 | DPO 在创作类任务上更贴近输出形式 |
| DPO 形式改善 | `manual_2196` 10 句话科技趋势总结 | 以段落总结为主，没有严格按 10 句组织 | 使用编号列表，覆盖 AI、5G、区块链、云计算等趋势 | DPO 的结构化输出更明显，但仍被长度上限截断 |
| DPO 形式改善 | `manual_2284` 描述公园景色 | 回答较泛，末尾要求用户补充更多信息 | 直接描写花坛、树木、小径、水池等景物 | DPO 更贴近“描述景色”的任务 |
| DPO 相对改善但有幻觉 | `manual_2949` 推荐餐厅和招牌菜 | 不合理拒答，未完成推荐任务 | 给出餐厅和招牌菜 | DPO 更愿意执行指令，但餐厅和菜品真实性不可靠 |
| DPO 退化 | `manual_8244` 内部沟通方案 | 给出完整 6 点分析和改进步骤 | 只列到第 3 点且截断 | DPO 更短，但完整性下降 |
| DPO 退化 | `manual_3223` 日期抽取 | 输出 `2022-03-01`、`2022-04-30`，基本完成抽取 | 编造更多课程日期，没有只返回日期列表 | DPO 在抽取任务上偏离指令 |
| DPO 退化 | `manual_5217` 排队等候对话 | 虽然主题有些发散，但仍围绕排队/交通等待 | 对话变成会面邀请，偏离“排队等候” | DPO 指令遵循下降 |
| 两者都失败，DPO 更严重 | `manual_1849` 太阳系概括 | 有不严谨内容，但基本知道太阳系由行星、卫星等组成 | 出现“人造天体系统”“地球是太阳系中心”等明显事实错误 | DPO 未改善事实可靠性，甚至退化 |
| 两者都失败 | `manual_5701` iPhone Touch ID 设置 | 步骤不准确，如“应用菜单”“解锁按钮” | 编造 Touch ID 应用和错误入口 | 生活操作类问题仍有幻觉 |
| 两者都不理想 | `manual_7225` 房价预测 | 给出方法和代码开头，但被截断，没有最终 12 个月预测 | 只给出一个数值 `3700`，缺少方法和 12 个月序列 | DPO 更简洁但信息不足 |

综合观察：

- DPO 输出整体更短，部分任务更像目标格式，例如诗歌、编号趋势总结和景色描写。
- DPO 并没有稳定提升指令遵循。日期抽取、对话生成、事实问答和操作指导中都能看到退化。
- DPO 后的输出更容易出现早停或截断，尤其在需要长答案、代码或多步骤方案的任务上。
- fixed prompt 的人工分析与自动指标一致：DPO 对偏好排序有局部影响，但不能证明通用问答质量提升。

## Preference Accuracy

评估目标：比较 SFT adapter 和 DPO adapter 对 `preference_test_100` 中 `chosen/rejected` 的偏好。每条样本分别计算 chosen 和 rejected 在同一模型下的平均 answer token logprob，并统计 `chosen_logprob > rejected_logprob` 的比例。

输出：

```text
results/preference_accuracy_sft_dpo_qwen05b.json
results/preference_accuracy_sft_dpo_qwen05b_records.jsonl
```

结果：

| 模型 | chosen preferred | rejected preferred | ties | accuracy | mean margin |
|---|---:|---:|---:|---:|---:|
| SFT-3k adapter | 7 / 100 | 93 / 100 | 0 | 0.07 | -0.7120 |
| DPO-300 adapter | 13 / 100 | 87 / 100 | 0 | 0.13 | -0.5464 |

观察：

- DPO-300 的 preference accuracy 从 SFT 的 0.07 提升到 0.13，mean margin 从 -0.7120 改善到 -0.5464，说明 DPO 后模型相对更偏向 chosen。
- 但绝对 accuracy 仍然很低，说明模型多数情况下仍给 rejected 更高平均 logprob。
- 一个可能原因是 rejected 来自 SFT adapter 自身生成，风格上更接近当前模型分布；BELLE reference 虽然作为 chosen，但不一定在模型概率空间中更容易被赋高分。

## 当前结论

当前最稳妥的结论是：

> 在 300 条偏好样本的小规模离线 DPO 实验中，DPO-300 相比 SFT-3k 在 preference accuracy 上有轻微改善，但在原 `test_100` 的 ROUGE-L、eval loss 和 perplexity 上没有提升。因此该阶段只能说明 DPO 对 chosen/rejected 排序倾向产生了一定影响，不能夸大为通用能力提升或完整 RLHF 对齐成功。

## 后续建议

1. 如继续扩展，可尝试更高质量的 rejected 构造方式，例如人工筛选失败回答或使用规则过滤明显错误答案。
2. 可以增加更短的 `max_new_tokens` / 更长的 `max_new_tokens` 对比，确认 DPO 输出偏短是模型倾向还是生成长度设置影响。
3. README 中保持保守表达，把 DPO 结果作为“小规模偏好对齐实验现象”，不要包装成生产级对齐结论。
