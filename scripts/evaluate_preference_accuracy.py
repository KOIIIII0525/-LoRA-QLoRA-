"""Evaluate whether adapters assign higher logprob to chosen than rejected."""

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
from src.preference_eval_utils import build_preference_score_record, summarize_preference_scores
from src.preference_utils import validate_preference_row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT/DPO chosen-vs-rejected preference accuracy.")
    parser.add_argument("--config", type=str, default="configs/dpo_qwen_0.5b.yaml", help="Path to DPO config.")
    parser.add_argument("--preference_file", type=str, default=None, help="Preference test JSONL override.")
    parser.add_argument("--sft_adapter_dir", type=str, default=None, help="SFT adapter directory override.")
    parser.add_argument("--dpo_adapter_dir", type=str, default=None, help="DPO adapter directory override.")
    parser.add_argument("--output_file", type=str, default=None, help="Metrics JSON output path.")
    parser.add_argument("--records_file", type=str, default=None, help="Optional per-sample JSONL output path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max records for quick checks.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without loading models.")
    return parser.parse_args(argv)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def resolve_preference_eval_inputs(
    config: dict[str, Any],
    preference_file: str | None,
    sft_adapter_dir: str | None,
    dpo_adapter_dir: str | None,
    output_file: str | None,
) -> dict[str, Path]:
    preference_outputs = config["preference"]["output_files"]
    return {
        "preference_file": Path(preference_file or preference_outputs["preference_test_100"]),
        "sft_adapter_dir": Path(sft_adapter_dir or config["model"]["sft_adapter_dir"]),
        "dpo_adapter_dir": Path(dpo_adapter_dir or config["dpo"]["output_dir"]),
        "output_file": Path(output_file or "results/preference_accuracy_sft_dpo_qwen05b.json"),
    }


def resolve_records_file(records_file: str | None) -> Path | None:
    return Path(records_file) if records_file else None


def load_preference_rows(path: str | Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, row in enumerate(load_jsonl(path)):
        if not isinstance(row, dict) or not validate_preference_row(row):
            raise ValueError(f"Invalid preference row at index {index}")
        rows.append(
            {
                "id": str(row.get("id", f"pref_{index}")),
                "prompt": str(row["prompt"]),
                "chosen": str(row["chosen"]),
                "rejected": str(row["rejected"]),
            }
        )
    if limit is not None:
        rows = rows[:limit]
    return rows


def import_scoring_dependencies() -> dict[str, Any]:
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


def load_adapter_model(config: dict[str, Any], adapter_dir: str | Path, deps: dict[str, Any]):
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
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer, torch


def mean_answer_logprob(model: Any, tokenizer: Any, torch: Any, prompt: str, answer: str) -> float:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(prompt_text + answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    if full_ids.shape[-1] <= prompt_len:
        raise ValueError("Answer produced no scoreable tokens")

    input_ids = full_ids.to(model.device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if int(mask.sum().detach().cpu()) == 0:
        raise ValueError("No answer tokens remained after masking")

    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    gathered = log_probs.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    answer_log_probs = gathered[mask]
    return float(answer_log_probs.mean().detach().cpu())


def score_rows_for_adapter(
    config: dict[str, Any],
    adapter_dir: str | Path,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    deps = import_scoring_dependencies()
    model, tokenizer, torch = load_adapter_model(config, adapter_dir, deps)
    try:
        scored_rows = []
        for row in rows:
            chosen_logprob = mean_answer_logprob(model, tokenizer, torch, row["prompt"], row["chosen"])
            rejected_logprob = mean_answer_logprob(model, tokenizer, torch, row["prompt"], row["rejected"])
            scored_rows.append(
                build_preference_score_record(
                    sample_id=row["id"],
                    prompt=row["prompt"],
                    chosen=row["chosen"],
                    rejected=row["rejected"],
                    chosen_logprob=chosen_logprob,
                    rejected_logprob=rejected_logprob,
                )
            )
        return scored_rows
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resolved = resolve_preference_eval_inputs(
        config,
        args.preference_file,
        args.sft_adapter_dir,
        args.dpo_adapter_dir,
        args.output_file,
    )
    rows = load_preference_rows(resolved["preference_file"], args.limit)
    records_file = resolve_records_file(args.records_file)

    if args.dry_run:
        print("PREFERENCE_ACCURACY_DRY_RUN_OK")
        print(f"records: {len(rows)}")
        print(f"preference_file: {resolved['preference_file']}")
        print(f"sft_adapter_dir: {resolved['sft_adapter_dir']}")
        print(f"dpo_adapter_dir: {resolved['dpo_adapter_dir']}")
        print(f"output_file: {resolved['output_file']}")
        if records_file:
            print(f"records_file: {records_file}")
        return

    sft_records = score_rows_for_adapter(config, resolved["sft_adapter_dir"], rows)
    dpo_records = score_rows_for_adapter(config, resolved["dpo_adapter_dir"], rows)
    metrics = {
        "num_examples": len(rows),
        "scoring": "mean_answer_token_logprob",
        "sft": summarize_preference_scores(sft_records),
        "dpo": summarize_preference_scores(dpo_records),
    }
    save_json(resolved["output_file"], metrics)

    if records_file:
        output_rows = []
        for index, row in enumerate(rows):
            output_rows.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                    "sft": {
                        "chosen_logprob": sft_records[index]["chosen_logprob"],
                        "rejected_logprob": sft_records[index]["rejected_logprob"],
                        "margin": sft_records[index]["margin"],
                        "preferred": sft_records[index]["preferred"],
                    },
                    "dpo": {
                        "chosen_logprob": dpo_records[index]["chosen_logprob"],
                        "rejected_logprob": dpo_records[index]["rejected_logprob"],
                        "margin": dpo_records[index]["margin"],
                        "preferred": dpo_records[index]["preferred"],
                    },
                }
            )
        write_jsonl(records_file, output_rows)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
