"""Utilities for lightweight DPO alignment training."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import yaml


def load_dpo_config(config_path: str | Path) -> dict[str, Any]:
    """Load a DPO YAML config."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def count_jsonl_rows(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_first_jsonl_row(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No non-empty rows found in {path}")


def resolve_dpo_inputs(
    config: dict[str, Any],
    train_split: str,
    output_dir: str | Path | None,
) -> dict[str, Path]:
    """Resolve DPO train/eval/output paths from config and CLI overrides."""
    preference_config = config["preference"]
    output_files = preference_config["output_files"]
    if train_split not in output_files:
        raise KeyError(f"Unknown DPO train split '{train_split}'. Available splits: {sorted(output_files)}")

    default_output_dir = (
        config["dpo"]["smoke_output_dir"]
        if train_split == "preference_train_50"
        else config["dpo"]["output_dir"]
    )

    return {
        "train_file": Path(output_files[train_split]),
        "eval_file": Path(output_files["preference_valid_100"]),
        "output_dir": Path(output_dir) if output_dir else Path(default_output_dir),
        "sft_adapter_dir": Path(config["model"]["sft_adapter_dir"]),
    }


def validate_preference_schema(row: Any, path: Path) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"Expected preference dict row in {path}, got {type(row).__name__}")
    for key in ("prompt", "chosen", "rejected"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Preference row in {path} is missing non-empty '{key}'")
    if row["chosen"].strip() == row["rejected"].strip():
        raise ValueError(f"Preference row in {path} has identical chosen/rejected")


def validate_dpo_inputs(
    config: dict[str, Any],
    resolved: dict[str, Path],
) -> dict[str, Any]:
    """Validate DPO data/model paths and return a compact summary."""
    train_file = resolved["train_file"]
    eval_file = resolved["eval_file"]
    sft_adapter_dir = resolved["sft_adapter_dir"]

    if not train_file.exists():
        raise FileNotFoundError(f"DPO train file not found: {train_file}")
    if not sft_adapter_dir.exists():
        raise FileNotFoundError(f"SFT adapter dir not found: {sft_adapter_dir}")

    first_train_row = read_first_jsonl_row(train_file)
    validate_preference_schema(first_train_row, train_file)

    eval_rows = 0
    if eval_file.exists():
        first_eval_row = read_first_jsonl_row(eval_file)
        validate_preference_schema(first_eval_row, eval_file)
        eval_rows = count_jsonl_rows(eval_file)

    return {
        "train_rows": count_jsonl_rows(train_file),
        "eval_rows": eval_rows,
        "max_length": int(config["dpo"]["max_length"]),
        "max_prompt_length": int(config["dpo"]["max_prompt_length"]),
    }


def build_dpo_dry_run_report(
    config: dict[str, Any],
    resolved: dict[str, Path],
    summary: dict[str, Any],
) -> str:
    """Build a readable DPO dry-run report without importing heavy training deps."""
    model_config = config["model"]
    dpo_config = config["dpo"]

    lines = [
        "DPO training dry run",
        f"base_model: {model_config['base_model_name_or_path']}",
        f"sft_adapter_dir: {resolved['sft_adapter_dir']}",
        f"use_qlora: {model_config.get('use_qlora', False)}",
        f"load_in_4bit: {model_config.get('load_in_4bit', False)}",
        f"train_file: {resolved['train_file']}",
        f"eval_file: {resolved['eval_file']}",
        f"output_dir: {resolved['output_dir']}",
        f"train_rows: {summary['train_rows']}",
        f"eval_rows: {summary['eval_rows']}",
        f"beta: {dpo_config['beta']}",
        f"max_length: {summary['max_length']}",
        f"max_prompt_length: {summary['max_prompt_length']}",
        f"batch_size: {dpo_config['per_device_train_batch_size']}",
        f"grad_accumulation: {dpo_config['gradient_accumulation_steps']}",
        f"learning_rate: {dpo_config['learning_rate']}",
    ]
    return "\n".join(lines)


def build_dpo_arguments_kwargs(
    config: dict[str, Any],
    output_dir: str | Path,
    dpo_config_cls: Any,
    has_eval_dataset: bool = True,
) -> dict[str, Any]:
    """Build DPOConfig kwargs while tolerating TRL/Transformers API differences."""
    dpo_config = config["dpo"]
    kwargs = {
        "output_dir": str(output_dir),
        "beta": float(dpo_config["beta"]),
        "num_train_epochs": float(dpo_config["num_train_epochs"]),
        "per_device_train_batch_size": int(dpo_config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(dpo_config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(dpo_config["gradient_accumulation_steps"]),
        "learning_rate": float(dpo_config["learning_rate"]),
        "warmup_ratio": float(dpo_config["warmup_ratio"]),
        "logging_steps": int(dpo_config["logging_steps"]),
        "eval_steps": int(dpo_config["eval_steps"]),
        "save_steps": int(dpo_config["save_steps"]),
        "save_total_limit": int(dpo_config["save_total_limit"]),
        "fp16": bool(dpo_config["fp16"]),
        "bf16": False,
        "gradient_checkpointing": bool(dpo_config["gradient_checkpointing"]),
        "max_length": int(dpo_config["max_length"]),
        "max_prompt_length": int(dpo_config["max_prompt_length"]),
        "save_strategy": "steps",
        "report_to": [],
    }

    parameters = inspect.signature(dpo_config_cls.__init__).parameters
    eval_value = "steps" if has_eval_dataset else "no"
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = eval_value
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = eval_value

    return {key: value for key, value in kwargs.items() if key in parameters}
