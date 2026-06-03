"""Train a lightweight DPO adapter after QLoRA SFT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_jsonl
from src.dpo_utils import (
    build_dpo_arguments_kwargs,
    build_dpo_dry_run_report,
    load_dpo_config,
    resolve_dpo_inputs,
    validate_dpo_inputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen DPO adapter from SFT preference data.")
    parser.add_argument("--config", type=str, default="configs/dpo_qwen_0.5b.yaml", help="Path to DPO YAML config.")
    parser.add_argument(
        "--train_split",
        type=str,
        default="preference_train_50",
        choices=["preference_train_50", "preference_train_300"],
        help="Preference training split from config.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory override.")
    parser.add_argument("--dry_run", action="store_true", help="Validate config/data without loading models.")
    parser.add_argument("--check_env", action="store_true", help="Check heavy DPO dependencies.")
    return parser.parse_args(argv)


def import_dpo_dependencies() -> dict[str, Any]:
    """Import heavy DPO dependencies only for env checks or real training."""
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    return {
        "torch": torch,
        "Dataset": Dataset,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
    }


def build_dpo_dataset(rows: list[dict[str, str]], tokenizer: Any) -> list[dict[str, str]]:
    """Format preference rows for TRL DPOTrainer."""
    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_rows.append(
            {
                "prompt": prompt_text,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )
    return formatted_rows


def load_policy_model_and_tokenizer(config: dict[str, Any], resolved: dict[str, Path], deps: dict[str, Any]):
    """Load the quantized base model and trainable SFT adapter."""
    torch = deps["torch"]
    PeftModel = deps["PeftModel"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]

    model_config = config["model"]
    base_model_path = model_config["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if model_config.get("use_qlora", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=bool(model_config.get("load_in_4bit", True)),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    if model_config.get("use_qlora", False):
        model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, resolved["sft_adapter_dir"], is_trainable=True)
    return model, tokenizer


def run_training(config: dict[str, Any], resolved: dict[str, Path]) -> None:
    try:
        deps = import_dpo_dependencies()
    except ImportError as exc:
        raise SystemExit(
            "DPO dependencies are missing. Install requirements in the training environment "
            "before running DPO. Dry-run works without these deps."
        ) from exc

    Dataset = deps["Dataset"]
    DPOConfig = deps["DPOConfig"]
    DPOTrainer = deps["DPOTrainer"]

    model, tokenizer = load_policy_model_and_tokenizer(config, resolved, deps)
    train_dataset = Dataset.from_list(build_dpo_dataset(load_jsonl(resolved["train_file"]), tokenizer))
    eval_dataset = None
    if resolved["eval_file"].exists():
        eval_dataset = Dataset.from_list(build_dpo_dataset(load_jsonl(resolved["eval_file"]), tokenizer))

    dpo_args = DPOConfig(
        **build_dpo_arguments_kwargs(
            config=config,
            output_dir=resolved["output_dir"],
            dpo_config_cls=DPOConfig,
            has_eval_dataset=eval_dataset is not None,
        )
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(resolved["output_dir"]))


def main() -> None:
    args = parse_args()
    config = load_dpo_config(args.config)
    resolved = resolve_dpo_inputs(config, train_split=args.train_split, output_dir=args.output_dir)
    summary = validate_dpo_inputs(config, resolved)

    if args.check_env:
        try:
            deps = import_dpo_dependencies()
        except ImportError as exc:
            raise SystemExit(f"DPO_ENV_CHECK_FAILED: {exc}") from exc
        torch = deps["torch"]
        print("DPO_ENV_CHECK_OK")
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"device: {torch.cuda.get_device_name(0)}")
        return

    if args.dry_run:
        print(build_dpo_dry_run_report(config, resolved, summary))
        return

    run_training(config, resolved)


if __name__ == "__main__":
    main()
