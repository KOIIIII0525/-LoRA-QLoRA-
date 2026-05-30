"""Training utilities for LoRA/QLoRA SFT."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

import yaml


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML training config."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def count_jsonl_rows(path: str | Path) -> int:
    """Count non-empty JSONL rows."""
    with Path(path).open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_first_jsonl_row(path: str | Path) -> Any:
    """Read the first non-empty JSONL row."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No non-empty rows found in {path}")


def resolve_training_inputs(
    config: dict[str, Any],
    train_split: str,
    output_dir: str | Path | None,
) -> dict[str, Path]:
    """Resolve train/eval/output paths from config and CLI overrides."""
    data_config = config["data"]
    training_config = config["training"]

    if train_split not in data_config:
        raise KeyError(f"Unknown train split '{train_split}'. Available data keys: {sorted(data_config)}")

    return {
        "train_file": Path(data_config[train_split]),
        "eval_file": Path(data_config["valid_300"]),
        "output_dir": Path(output_dir) if output_dir else Path(training_config["output_dir"]),
    }


def validate_training_inputs(
    config: dict[str, Any],
    resolved: dict[str, Path],
) -> dict[str, Any]:
    """Validate paths and return a compact training input summary."""
    train_file = resolved["train_file"]
    eval_file = resolved["eval_file"]

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    first_train_row = read_first_jsonl_row(train_file)
    if not isinstance(first_train_row, list):
        raise ValueError(f"Expected chat messages list in {train_file}, got {type(first_train_row).__name__}")

    return {
        "train_rows": count_jsonl_rows(train_file),
        "eval_rows": count_jsonl_rows(eval_file),
        "max_seq_len": int(config["data"]["max_seq_len"]),
        "first_train_turns": len(first_train_row),
    }


def build_dry_run_report(
    config: dict[str, Any],
    resolved: dict[str, Path],
    summary: dict[str, Any],
) -> str:
    """Build a readable dry-run report without importing heavy training deps."""
    model_config = config["model"]
    training_config = config["training"]
    lora_config = config["lora"]

    lines = [
        "LoRA/QLoRA training dry run",
        f"base_model: {model_config['base_model_name_or_path']}",
        f"use_qlora: {model_config.get('use_qlora', False)}",
        f"load_in_4bit: {model_config.get('load_in_4bit', False)}",
        f"train_file: {resolved['train_file']}",
        f"eval_file: {resolved['eval_file']}",
        f"output_dir: {resolved['output_dir']}",
        f"train_rows: {summary['train_rows']}",
        f"eval_rows: {summary['eval_rows']}",
        f"max_seq_len: {summary['max_seq_len']}",
        f"batch_size: {training_config['per_device_train_batch_size']}",
        f"grad_accumulation: {training_config['gradient_accumulation_steps']}",
        f"learning_rate: {training_config['learning_rate']}",
        f"lora_r: {lora_config['r']}",
        f"lora_alpha: {lora_config['alpha']}",
    ]
    return "\n".join(lines)


def build_training_arguments_kwargs(
    config: dict[str, Any],
    output_dir: str | Path,
    training_arguments_cls: Any,
) -> dict[str, Any]:
    """Build TrainingArguments kwargs while tolerating transformers API renames."""
    training_config = config["training"]
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training_config["num_train_epochs"]),
        "per_device_train_batch_size": int(training_config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training_config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training_config["gradient_accumulation_steps"]),
        "learning_rate": float(training_config["learning_rate"]),
        "warmup_ratio": float(training_config["warmup_ratio"]),
        "weight_decay": float(training_config["weight_decay"]),
        "logging_steps": int(training_config["logging_steps"]),
        "eval_steps": int(training_config["eval_steps"]),
        "save_steps": int(training_config["save_steps"]),
        "save_total_limit": int(training_config["save_total_limit"]),
        "fp16": bool(training_config["fp16"]),
        "gradient_checkpointing": bool(training_config["gradient_checkpointing"]),
        "save_strategy": "steps",
        "report_to": [],
    }

    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return kwargs


def build_trainer_tokenizer_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    """Build Trainer tokenizer kwargs across transformers versions."""
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}
