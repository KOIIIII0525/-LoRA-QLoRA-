# 典型错误记录

用于记录项目中有代表性的错误，方便后续复盘和排查同类问题。

## 2026-05-28

### 1. `code/` 目录与 Python 标准库模块重名

- 现象：运行 `pytest` 时，`pdb` 导入标准库 `code`，却误导入了项目里的 `code/__init__.py`，进一步触发 `torch` 缺失错误。
- 原因：项目历史目录名 `code/` 与 Python 标准库 `code` 模块重名。
- 处理：先改用 `unittest` 避免 pytest 的导入链；随后把旧 `code/` 目录重命名为 `tiny_llm_legacy/`，根因解除。
- 复盘点：工程化项目中目录命名会影响 Python import 解析，旧教程代码需要隔离。

### 2. Windows / 沙箱临时目录权限问题

- 现象：测试使用系统临时目录或清理临时目录时出现 `PermissionError: WinError 5`。
- 原因：当前环境对用户临时目录和部分删除操作有限制，且 Windows 下文件锁更敏感。
- 处理：测试临时文件改写到仓库内 `tests/.tmp/`，并在 `.gitignore` 中忽略。
- 复盘点：测试不仅要验证逻辑，也要适配真实运行环境的文件系统约束。

### 3. 直接运行脚本时找不到 `src`

- 现象：执行 `python scripts/prepare_sft_data.py` 报 `ModuleNotFoundError: No module named 'src'`。
- 原因：直接运行脚本时，Python 默认把 `scripts/` 当作导入起点，项目根目录不一定在 `sys.path`。
- 处理：在脚本启动时把项目根目录加入 `sys.path`。
- 复盘点：命令行脚本要考虑用户从项目根目录直接运行的复现路径。

### 4. 补丁修改后出现 `try/except` 结构语法错误

- 现象：`scripts/train_lora.py` dry-run 报 `SyntaxError: expected 'except' or 'finally' block`。
- 原因：修改训练脚本时，把 `from src.data_utils import load_jsonl` 插入到了 `try` 代码块中间，破坏了缩进和结构。
- 处理：把轻量项目内导入放到文件顶部，保留重依赖导入只在训练函数内部懒加载。
- 复盘点：每次修改后要做导入检查和 dry-run，能快速抓住语法级和路径级问题。

### 5. 当前默认 Python 3.13 不适合作为训练环境

- 现象：当前系统 Python 是 3.13.9，且当前环境缺少 `torch` 等训练依赖。
- 原因：深度学习生态对 Python 3.10/3.11 支持更稳，QLoRA 相关依赖在 Windows 原生环境也可能不稳定。
- 处理：当前阶段只在 Python 3.13 中做数据处理、单元测试和 dry-run；真实训练切换到 Python 3.10/3.11 + PyTorch/Transformers/PEFT 环境。
- 复盘点：工程复现要区分“数据处理环境”和“训练环境”，不能把环境问题误判为算法问题。

### 6. `pytorch_env` 在项目根目录导入 `torch` 失败

- 现象：`D:\anaconda3\envs\pytorch_env\python.exe` 在项目根目录导入 `torch` 报循环导入错误；在项目目录外导入正常。
- 原因：项目根目录下的旧 `code/` 文件夹遮蔽了 Python 标准库 `code` 模块，PyTorch 导入链中的 `pdb -> code` 被错误解析。
- 处理：使用提升权限把旧目录重命名为 `tiny_llm_legacy/`，并删除失败迁移留下的缓存空壳目录；之后在项目根目录导入 `torch` 正常。
- 复盘点：训练环境本身可用，不代表项目根目录可复现；目录命名也会影响依赖导入。

### 7. `pytorch_env` 不是完整 LoRA/QLoRA 训练环境

- 现象：`pytorch_env` 有 PyTorch 2.10.0+cu126 且 CUDA 可用，但训练环境检查停在 `No module named 'peft'`。
- 原因：该环境主要是 PyTorch 环境，还缺少 LoRA/QLoRA 相关依赖。
- 处理：使用清华 PyPI 镜像安装 `peft`、`accelerate`、`bitsandbytes`、`sentencepiece`、`evaluate`、`rouge-score` 等依赖；随后 `--check_env` 通过。
- 复盘点：能跑 PyTorch 不等于能跑微调项目，训练前要检查完整依赖链。

### 8. Hugging Face 模型访问被拒导致 smoke test 卡住

- 现象：准备运行 `Qwen/Qwen2.5-0.5B-Instruct` 的 100 条训练冒烟测试时，当前环境访问 Hugging Face 模型仓库失败。未清理代理时是 `WinError 10061`，清理代理后 Hugging Face 原站变为连接超时。
- 原因：Shell 环境中存在指向 `127.0.0.1:9` 的代理变量；清理代理后，当前网络仍无法稳定直连 Hugging Face 原站。
- 处理：使用 `HF_ENDPOINT=https://hf-mirror.com` 访问国内镜像，并把模型完整下载到 `models/qwen2.5-0.5b-instruct/`；训练和评估配置已切换为本地模型路径。
- 复盘点：LLM 项目复现不仅依赖代码和显卡，也依赖模型权重获取方式；README 和 setup 文档应记录可复现的下载路径和离线加载方案。

### 9. `transformers` 版本升级导致训练脚本 API 不兼容

- 现象：真实 smoke test 中模型和数据都加载成功，但先后报错：`TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'` 和 `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`。
- 原因：当前 `pytorch_env` 中 `transformers` 为 `5.8.0`，部分 API 参数名相较常见 4.x 示例发生变化：`evaluation_strategy` 改为 `eval_strategy`，`Trainer(tokenizer=...)` 改为 `Trainer(processing_class=...)`。
- 处理：在 `src/train_utils.py` 中增加签名检测兼容层，根据当前安装版本动态选择参数名；补充单元测试覆盖新版参数名。
- 复盘点：训练脚本不能只照搬教程参数，应该通过 dry-run、smoke test 和小范围兼容层保证不同环境下可复现。

## 2026-05-30

### 10. eval loss 下降不等于领域概念回答正确

- 现象：`train_3k` QLoRA adapter 已完成训练，验证集 `eval_loss` 从 1k 的约 `1.7662` 降到 3k 的约 `1.7448`；但在单条推理 prompt `请用三句话解释什么是 LoRA。` 中，模型将 LoRA 解释成无线通信中的 LoRa，而不是大模型微调中的 Low-Rank Adaptation。
- 原因：训练数据主要来自通用中文指令数据，小规模 SFT 可以改善通用回答分布和验证集 loss，但不保证模型掌握特定算法概念；同时 LoRA/LoRa 本身存在大小写相近、语义歧义的问题。
- 处理：将该样例保留为固定 prompt / 人工分析中的失败案例，不把 eval loss 下降直接包装成“模型能力显著提升”；第 5 阶段继续补 Base vs Adapter 对比、ROUGE-L 和人工分析。
- 复盘点：自动指标只能说明一部分问题。LLM 微调项目需要同时报告成功样例、失败样例和局限，尤其要区分“语言建模 loss 改善”和“真实指令遵循/领域知识能力提升”。

### 11. 默认训练 split 与默认输出目录不一致

- 现象：`scripts/train_lora.py` 默认 `train_split=train_100`，但训练配置默认输出到 `outputs/qwen05b_qlora_3k`。如果直接运行 README 中的主训练命令，可能用 100 条数据覆盖或污染 3k 主实验目录。
- 原因：早期为了 smoke test 方便，把 CLI 默认 split 设为 `train_100`；项目进入主实验阶段后，配置默认输出目录已经切到 3k 主实验。
- 处理：将训练 CLI 默认 split 调整为 `train_3k`，并在 README 主实验命令中显式写出 `--train_split train_3k --output_dir outputs/qwen05b_qlora_3k`；smoke test 命令继续显式指定 `train_100` 和 `outputs/smoke`。
- 复盘点：实验项目进入主实验阶段后，默认命令必须偏向复现主结果，避免 smoke test 便利性反过来污染正式实验产物。

### 12. 推理命令使用错误配置文件

- 现象：README 中的推理命令曾使用 `configs/lora_qwen_0.5b.yaml`，但 `scripts/infer.py` 在 `adapter` 或 `both` 模式下需要 `adapter_dir`，该字段只存在于 `configs/eval.yaml`。
- 原因：训练配置和评估配置职责分离后，README 中遗留了旧命令。
- 处理：README 推理命令改为 `python scripts/infer.py --config configs/eval.yaml --prompt "请解释什么是 LoRA" --mode adapter`。
- 复盘点：训练配置和推理/评估配置分离后，README 命令必须跟着更新；否则脚本本身可用，但用户照文档复现会失败。

### 13. 本地结果文件被忽略导致公开摘要不可见

- 现象：`.gitignore` 正确忽略了 `results/*.jsonl` 和 `results/*.json`，避免提交大量本地产物；但 README 又引用了这些文件，公开仓库中读者可能看不到原始结果。
- 原因：本地实验产物和轻量结果摘要没有分层。
- 处理：保留 `results/` 忽略规则，同时新增 `docs/result_samples.md`，摘录核心指标和少量固定 prompt 对比样例。
- 复盘点：大文件和本地结果不应直接提交，但 README 需要可见证据；可以用摘要文档连接本地实验和公开仓库内容。

### 14. Git 初始化失败留下 config.lock

- 现象：第一次执行 `git init` 时，写入 `.git/config` 失败并留下 `.git/config.lock`，后续重新执行 `git init` 报 `could not lock config file ... File exists`。
- 原因：当前运行环境对 `.git/` 写入存在沙箱权限限制，第一次初始化未完整结束。
- 处理：确认 `.git/config` 不存在且 `.git/config.lock` 是失败初始化留下的 36 字节锁文件后，删除该锁文件，并使用提升权限重新执行 `git init`。
- 复盘点：Git 初始化也可能受本地权限/沙箱影响。处理锁文件前应先确认它是失败残留，而不是另一个正在运行的 Git 进程持有的锁。

## 2026-06-02

### 15. DPO rejected response 一次性生成耗时长且中途不可观察

- 现象：尝试一次性为 500 条 preference prompt 生成 SFT adapter rejected responses 时，用户因运行时间过长中断。检查后发现后台 `python` 进程仍在运行，GPU 利用率约 99%，但目标 JSONL 尚不存在。
- 原因：`scripts/infer.py` 会先逐条生成所有 response，再一次性写入输出 JSONL；在 500 条、`max_new_tokens=128` 的设置下，本地 RTX 3060 Laptop 6GB 推理耗时很长，中途没有可观察进度文件。
- 处理：停止后台长任务，给 `scripts/infer.py` 增加 `--offset` 参数，与 `--limit` 配合按 50 条分批生成；先完成 `results/preference_rejected_chunk_000.jsonl`，再生成 `data/processed/preference_train_50.jsonl` 作为 DPO smoke 数据。
- 复盘点：长时间 LLM 推理任务要优先考虑可恢复、可观察和分批落盘。第一版 DPO 应先跑 50 条 smoke test，确认训练链路后再继续生成 300/500 条主实验数据。

### 16. DPO fp16 训练触发 bfloat16 GradScaler 错误

- 现象：`preference_train_50` DPO smoke test 在模型加载、数据 tokenization 和 Trainer 初始化后，训练第 1 个 step 报 `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`。
- 原因：当前 Windows + PyTorch 2.10.0+cu126 + Transformers 5.8.0 + TRL 1.5.1 + Accelerate 组合中，DPO 的 AMP/GradScaler 路径遇到了 bfloat16 梯度 unscale 不支持的问题。检查可训练 LoRA 参数 dtype 为 `float32`，说明不是 adapter 参数本身保存成 bfloat16，而是混合精度训练路径导致。
- 处理：将 `configs/dpo_qwen_0.5b.yaml` 中 DPO 阶段 `fp16` 改为 `false`，显式传 `bf16=false`，仍保留 4-bit QLoRA 加载；随后 50 条 DPO smoke test 成功完成并生成 `outputs/qwen05b_dpo_50/adapter_model.safetensors`。
- 复盘点：SFT 可用的 fp16 配置不一定能原样迁移到 DPO/TRL。新训练范式要先做 smoke test，并把精度、量化、Trainer 版本兼容问题独立记录。
