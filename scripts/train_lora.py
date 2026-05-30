"""Train a LoRA/QLoRA adapter for lightweight Chinese instruction tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_utils import (
    build_trainer_tokenizer_kwargs,
    build_training_arguments_kwargs,
    build_dry_run_report,
    load_training_config,
    resolve_training_inputs,
    validate_training_inputs,
)
from src.data_utils import load_jsonl


def import_training_dependencies():
    """Import heavy training dependencies only when real training or env checks run."""
    import torch
    from datasets import Dataset, DatasetDict
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    return {
        "torch": torch,
        "Dataset": Dataset,
        "DatasetDict": DatasetDict,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA/QLoRA adapter.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_qwen_0.5b.yaml",
        help="Path to YAML training config.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train_3k",
        choices=["train_100", "train_1k", "train_3k"],
        help="Training split from config data section.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config/data and print planned settings without loading model.",
    )
    parser.add_argument(
        "--check_env",
        action="store_true",
        help="Check whether heavy training dependencies can be imported.",
    )
    return parser.parse_args(argv)


def run_training(config: dict, resolved: dict[str, Path]) -> None:
    """Run actual training. Heavy dependencies are imported only here."""
    try:
        deps = import_training_dependencies()
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install requirements in a Python 3.10/3.11 "
            "environment before running real training. Dry-run works without these deps."
        ) from exc

    torch = deps["torch"]
    Dataset = deps["Dataset"]
    DatasetDict = deps["DatasetDict"]
    LoraConfig = deps["LoraConfig"]
    TaskType = deps["TaskType"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]
    DataCollatorForLanguageModeling = deps["DataCollatorForLanguageModeling"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    model_config = config["model"]
    data_config = config["data"]
    lora_config = config["lora"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model_name_or_path"],
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
        model_config["base_model_name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    if model_config.get("use_qlora", False):
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
    )
    model = get_peft_model(model, peft_config)

    dataset = DatasetDict(
        {
            "train": Dataset.from_list([{"messages": row} for row in load_jsonl(resolved["train_file"])]),
            "validation": Dataset.from_list([{"messages": row} for row in load_jsonl(resolved["eval_file"])]),
        }
    )

    def format_and_tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=int(data_config["max_seq_len"]),
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset["train"].column_names,
    )

    args = TrainingArguments(
        **build_training_arguments_kwargs(
            config=config,
            output_dir=resolved["output_dir"],
            training_arguments_cls=TrainingArguments,
        )
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        **build_trainer_tokenizer_kwargs(Trainer, tokenizer),
    )
    trainer.train()
    trainer.save_model(str(resolved["output_dir"]))


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    resolved = resolve_training_inputs(config, train_split=args.train_split, output_dir=args.output_dir)
    summary = validate_training_inputs(config, resolved)

    if args.check_env:
        try:
            deps = import_training_dependencies()
        except ImportError as exc:
            raise SystemExit(f"ENV_CHECK_FAILED: {exc}") from exc
        torch = deps["torch"]
        print("ENV_CHECK_OK")
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"device: {torch.cuda.get_device_name(0)}")
        return

    if args.dry_run:
        print(build_dry_run_report(config, resolved, summary))
        return

    run_training(config, resolved)


if __name__ == "__main__":
    main()
