# 环境安装与验证

本项目第一版使用本机 `pytorch_env` 作为训练环境。

## 已验证环境

| 项目 | 当前值 |
|---|---|
| Conda env | `D:\anaconda3\envs\pytorch_env` |
| Python | `3.12.12` |
| PyTorch | `2.10.0+cu126` |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6GB |

默认系统 Python `3.13.9` 只用于轻量脚本检查，不作为训练环境。

## 安装依赖

在 PowerShell 中使用：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

如果只补训练核心依赖：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple accelerate peft sentencepiece evaluate rouge-score scipy bitsandbytes
```

## 环境验证

检查 PyTorch 和 CUDA：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

检查训练依赖：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --check_env
```

预期输出包含：

```text
ENV_CHECK_OK
cuda_available: True
device: NVIDIA GeForce RTX 3060 Laptop GPU
```

## 模型下载

当前环境直接访问 Hugging Face 原站不稳定，第一版优先使用国内镜像下载 `Qwen/Qwen2.5-0.5B-Instruct` 到本地目录：

```powershell
$env:ALL_PROXY=''
$env:HTTP_PROXY=''
$env:HTTPS_PROXY=''
$env:GIT_HTTP_PROXY=''
$env:GIT_HTTPS_PROXY=''
$env:HF_ENDPOINT='https://hf-mirror.com'
D:\anaconda3\envs\pytorch_env\python.exe -B -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/qwen2.5-0.5b-instruct', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.py','*.tiktoken'])"
```

下载完成后，配置文件默认使用本地路径：

```yaml
model:
  base_model_name_or_path: models/qwen2.5-0.5b-instruct
```

训练前 dry-run：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --output_dir outputs\smoke --dry_run
```

## Smoke Test

100 条数据训练命令：

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --output_dir outputs\smoke
```

目标：

- 验证模型能加载。
- 验证 tokenizer 和 chat template 可用。
- 验证 6GB 显存下训练不会立即 OOM。
- 验证 LoRA/QLoRA adapter 可以保存。

## 注意事项

- 旧教程目录已改名为 `tiny_llm_legacy/`，避免遮蔽 Python 标准库 `code`。
- Windows 原生 QLoRA 依赖 `bitsandbytes`，如果真实训练失败，可先切到普通 LoRA 或使用 WSL2/Colab。
- `outputs/`、模型权重和大数据已在 `.gitignore` 中排除，不应提交到 Git。
