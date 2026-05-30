"""Run base-model and LoRA/QLoRA adapter inference."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_jsonl, write_jsonl
from src.eval_utils import build_prediction_record, normalize_prompt_record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen base/adapter inference.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="Path to eval config.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to generate.")
    parser.add_argument("--input_file", type=str, default=None, help="JSONL prompt file for batch inference.")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSONL path for batch predictions.")
    parser.add_argument("--mode", choices=["base", "adapter", "both"], default="both", help="Model variant to run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max records for batch inference.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override generation max_new_tokens.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without loading the model.")
    return parser.parse_args(argv)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def resolve_prompt_rows(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt:
        return [{"id": "prompt_0", "prompt": args.prompt, "reference": ""}]

    input_file = Path(args.input_file or config["data"]["manual_prompts_file"])
    rows = [normalize_prompt_record(row, index) for index, row in enumerate(load_jsonl(input_file))]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def build_generation_kwargs(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    generation = config.get("generation", {})
    return {
        "max_new_tokens": int(args.max_new_tokens or generation.get("max_new_tokens", 256)),
        "temperature": float(generation.get("temperature", 0.7)),
        "top_p": float(generation.get("top_p", 0.9)),
        "do_sample": bool(generation.get("do_sample", True)),
    }


def import_inference_dependencies():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def load_model_and_tokenizer(config: dict[str, Any], use_adapter: bool):
    deps = import_inference_dependencies()
    torch = deps["torch"]
    PeftModel = deps["PeftModel"]
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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
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
    if use_adapter:
        model = PeftModel.from_pretrained(model, model_config["adapter_dir"])
    model.eval()
    return model, tokenizer, torch


def generate_one(model: Any, tokenizer: Any, torch: Any, prompt: str, generation_kwargs: dict[str, Any]) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_variant_responses(
    config: dict[str, Any],
    rows: list[dict[str, str]],
    variant: str,
    generation_kwargs: dict[str, Any],
) -> list[str]:
    model, tokenizer, torch = load_model_and_tokenizer(config, use_adapter=(variant == "adapter"))
    try:
        return [generate_one(model, tokenizer, torch, row["prompt"], generation_kwargs) for row in rows]
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_output_records(
    rows: list[dict[str, str]],
    base_responses: list[str] | None,
    adapter_responses: list[str] | None,
) -> list[dict[str, str]]:
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            build_prediction_record(
                sample_id=row["id"],
                prompt=row["prompt"],
                reference=row["reference"],
                base_response=base_responses[index] if base_responses is not None else "",
                adapter_response=adapter_responses[index] if adapter_responses is not None else "",
            )
        )
    return output_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = resolve_prompt_rows(config, args)
    generation_kwargs = build_generation_kwargs(config, args)
    output_file = Path(args.output_file or "results/manual_compare_qwen05b_qlora_3k.jsonl")

    if args.dry_run:
        print("INFER_DRY_RUN_OK")
        print(f"mode: {args.mode}")
        print(f"records: {len(rows)}")
        print(f"output_file: {output_file}")
        print(f"max_new_tokens: {generation_kwargs['max_new_tokens']}")
        return

    base_responses = None
    adapter_responses = None
    if args.mode in {"base", "both"}:
        base_responses = generate_variant_responses(config, rows, "base", generation_kwargs)
    if args.mode in {"adapter", "both"}:
        adapter_responses = generate_variant_responses(config, rows, "adapter", generation_kwargs)

    output_rows = build_output_records(rows, base_responses, adapter_responses)
    if args.prompt:
        print(json.dumps(output_rows[0], ensure_ascii=False, indent=2))
    else:
        write_jsonl(output_file, output_rows)
        print(f"WROTE {len(output_rows)} records to {output_file}")


if __name__ == "__main__":
    main()
