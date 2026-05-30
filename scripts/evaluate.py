"""Evaluate QLoRA adapter predictions with lightweight automatic metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.infer import build_generation_kwargs, generate_variant_responses, load_config
from src.data_utils import load_jsonl, write_jsonl
from src.eval_utils import build_prediction_record, normalize_prompt_record, safe_perplexity, summarize_losses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen QLoRA adapter.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="Path to eval config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max records for quick checks.")
    parser.add_argument("--predictions_file", type=str, default=None, help="Optional predictions JSONL override.")
    parser.add_argument("--metrics_file", type=str, default=None, help="Optional metrics JSON override.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override generation max_new_tokens.")
    parser.add_argument("--skip_generation", action="store_true", help="Reuse existing predictions file.")
    parser.add_argument("--skip_perplexity", action="store_true", help="Skip eval loss / perplexity computation.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without loading the model.")
    return parser.parse_args(argv)


def resolve_eval_rows(config: dict[str, Any], limit: int | None) -> list[dict[str, str]]:
    rows = [normalize_prompt_record(row, index) for index, row in enumerate(load_jsonl(config["data"]["test_file"]))]
    if limit is not None:
        rows = rows[:limit]
    return rows


def apply_prediction_limit(prediction_rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return prediction_rows
    return prediction_rows[:limit]


def compute_rouge_l(prediction_rows: list[dict[str, str]]) -> dict[str, float | int]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for row in prediction_rows:
        reference = row.get("reference", "")
        prediction = row.get("adapter_response", "")
        if not reference or not prediction:
            continue
        scores.append(scorer.score(reference, prediction)["rougeL"].fmeasure)
    if not scores:
        return {"count": 0, "rouge_l": 0.0}
    return {"count": len(scores), "rouge_l": sum(scores) / len(scores)}


def import_eval_dependencies():
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


def load_adapter_model(config: dict[str, Any]):
    deps = import_eval_dependencies()
    torch = deps["torch"]
    PeftModel = deps["PeftModel"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]

    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model_name_or_path"],
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
        model_config["base_model_name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, model_config["adapter_dir"])
    model.eval()
    return model, tokenizer, torch


def compute_eval_loss(config: dict[str, Any], raw_rows: list[Any]) -> dict[str, float | int | None]:
    model, tokenizer, torch = load_adapter_model(config)
    max_seq_len = int(config.get("data", {}).get("max_seq_len", 512))
    losses: list[float | None] = []
    try:
        for row in raw_rows:
            if isinstance(row, list):
                messages = row
            else:
                normalized = normalize_prompt_record(row, len(losses))
                messages = [
                    {"role": "user", "content": normalized["prompt"]},
                    {"role": "assistant", "content": normalized["reference"]},
                ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            tokenized = {key: value.to(model.device) for key, value in tokenized.items()}
            labels = tokenized["input_ids"].clone()
            with torch.no_grad():
                output = model(**tokenized, labels=labels)
            losses.append(float(output.loss.detach().cpu()))
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_losses(losses)
    mean_loss = summary["mean"]
    return {
        "count": summary["count"],
        "eval_loss": mean_loss,
        "perplexity": safe_perplexity(float(mean_loss)) if mean_loss is not None else None,
    }


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = resolve_eval_rows(config, args.limit)
    raw_rows = load_jsonl(config["data"]["test_file"])
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]

    predictions_file = Path(args.predictions_file or config["outputs"]["predictions_file"])
    metrics_file = Path(args.metrics_file or config["outputs"]["metrics_file"])

    if args.dry_run:
        print("EVAL_DRY_RUN_OK")
        print(f"records: {len(rows)}")
        print(f"predictions_file: {predictions_file}")
        print(f"metrics_file: {metrics_file}")
        print(f"skip_generation: {args.skip_generation}")
        print(f"skip_perplexity: {args.skip_perplexity}")
        return

    if args.skip_generation:
        prediction_rows = apply_prediction_limit(load_jsonl(predictions_file), args.limit)
    else:
        adapter_responses = generate_variant_responses(
            config,
            rows,
            "adapter",
            build_generation_kwargs(config, argparse.Namespace(max_new_tokens=args.max_new_tokens)),
        )
        prediction_rows = [
            build_prediction_record(
                sample_id=row["id"],
                prompt=row["prompt"],
                reference=row["reference"],
                adapter_response=adapter_responses[index],
            )
            for index, row in enumerate(rows)
        ]
        write_jsonl(predictions_file, prediction_rows)

    metrics: dict[str, Any] = {
        "num_examples": len(rows),
        "rouge_l": compute_rouge_l(prediction_rows),
    }
    if not args.skip_perplexity:
        metrics["perplexity"] = compute_eval_loss(config, raw_rows)
    else:
        metrics["perplexity"] = None

    save_metrics(metrics_file, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
