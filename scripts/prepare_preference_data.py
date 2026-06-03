"""Prepare lightweight preference data for DPO alignment experiments."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import has_user_and_assistant, load_jsonl, write_jsonl
from src.eval_utils import normalize_prompt_record
from src.preference_utils import build_preference_row, extract_prompt_reference, validate_preference_row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DPO preference prompts or chosen/rejected pairs.")
    parser.add_argument("--config", type=str, default="configs/dpo_qwen_0.5b.yaml", help="Path to DPO config.")
    parser.add_argument("--stage", choices=["prompts", "pairs"], default="prompts", help="Preparation stage.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for quick checks.")
    parser.add_argument("--only_split", type=str, default=None, help="Only build one configured preference split.")
    parser.add_argument("--predictions_file", type=str, default=None, help="Override rejected predictions JSONL.")
    parser.add_argument("--output_file", type=str, default=None, help="Override output JSONL for --only_split.")
    parser.add_argument("--dry_run", action="store_true", help="Validate settings without writing files.")
    return parser.parse_args(argv)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def collect_excluded_prompts(exclude_prompt_files: list[str | Path]) -> set[str]:
    excluded: set[str] = set()
    for path in exclude_prompt_files:
        file_path = Path(path)
        if not file_path.exists():
            continue
        for index, row in enumerate(load_jsonl(file_path)):
            normalized = normalize_prompt_record(row, index)
            excluded.add(normalized["prompt"])
    return excluded


def build_preference_prompt_pool(
    source_file: str | Path,
    output_file: str | Path,
    seed: int,
    pool_size: int,
    exclude_prompt_files: list[str | Path],
) -> Path:
    """Extract a deterministic prompt/reference pool for preference-response generation."""
    excluded_prompts = collect_excluded_prompts(exclude_prompt_files)
    candidates: list[dict[str, str]] = []
    for source_index, row in enumerate(load_jsonl(source_file)):
        if not has_user_and_assistant(row):
            continue
        prompt, reference = extract_prompt_reference(row)
        if not prompt or not reference or prompt in excluded_prompts:
            continue
        candidates.append(
            {
                "id": f"pref_prompt_{source_index}",
                "prompt": prompt,
                "reference": reference,
            }
        )

    if len(candidates) < pool_size:
        raise ValueError(f"Need at least {pool_size} preference prompt candidates, got {len(candidates)}")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    output_path = Path(output_file)
    write_jsonl(output_path, candidates[:pool_size])
    return output_path


def build_preference_pairs_from_predictions(
    predictions_file: str | Path,
    output_file: str | Path,
    limit: int | None,
    rejected_field: str,
) -> Path:
    """Convert inference predictions to chosen/rejected DPO preference rows."""
    rows = load_jsonl(predictions_file)
    if limit is not None:
        rows = rows[:limit]

    preference_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        preference_row = build_preference_row(
            sample_id=str(row.get("id", f"pref_{index}")).replace("pref_prompt_", "pref_"),
            prompt=str(row.get("prompt", "")),
            chosen=str(row.get("reference", "")),
            rejected=str(row.get(rejected_field, "")),
            source=f"reference_vs_{rejected_field}",
        )
        if validate_preference_row(preference_row):
            preference_rows.append(preference_row)

    output_path = Path(output_file)
    write_jsonl(output_path, preference_rows)
    return output_path


def build_all_preference_splits_from_predictions(
    predictions_file: str | Path,
    output_files: dict[str, str | Path],
    train_split_sizes: dict[str, int],
    eval_split_sizes: dict[str, int],
    rejected_field: str,
) -> dict[str, Path]:
    """Build nested train splits plus disjoint validation/test preference splits."""
    all_rows = load_jsonl(predictions_file)
    preference_rows: list[dict[str, str]] = []
    for index, row in enumerate(all_rows):
        preference_row = build_preference_row(
            sample_id=str(row.get("id", f"pref_{index}")).replace("pref_prompt_", "pref_"),
            prompt=str(row.get("prompt", "")),
            chosen=str(row.get("reference", "")),
            rejected=str(row.get(rejected_field, "")),
            source=f"reference_vs_{rejected_field}",
        )
        if validate_preference_row(preference_row):
            preference_rows.append(preference_row)

    max_train_size = max(train_split_sizes.values(), default=0)
    required = max_train_size + sum(eval_split_sizes.values())
    if len(preference_rows) < required:
        raise ValueError(f"Need at least {required} valid preference rows, got {len(preference_rows)}")

    written: dict[str, Path] = {}
    for split_name, size in train_split_sizes.items():
        output_path = Path(output_files[split_name])
        write_jsonl(output_path, preference_rows[: int(size)])
        written[split_name] = output_path

    cursor = max_train_size
    for split_name, size in eval_split_sizes.items():
        output_path = Path(output_files[split_name])
        split_rows = preference_rows[cursor : cursor + int(size)]
        cursor += int(size)
        write_jsonl(output_path, split_rows)
        written[split_name] = output_path

    return written


def required_prompt_pool_size(config: dict[str, Any], limit: int | None) -> int:
    preference_config = config["preference"]
    train_sizes = preference_config.get("train_split_sizes", {})
    eval_sizes = preference_config.get("eval_split_sizes", {})
    default_size = max(train_sizes.values(), default=0) + sum(eval_sizes.values())
    configured_size = int(preference_config.get("prompt_pool_size", default_size))
    return min(configured_size, limit) if limit is not None else configured_size


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"].get("seed", 42))
    preference_config = config["preference"]

    if args.stage == "prompts":
        pool_size = required_prompt_pool_size(config, args.limit)
        output_file = Path(preference_config["prompt_pool_file"])
        if args.dry_run:
            print("PREFERENCE_PROMPTS_DRY_RUN_OK")
            print(f"source_file: {preference_config['source_file']}")
            print(f"output_file: {output_file}")
            print(f"pool_size: {pool_size}")
            return
        written = build_preference_prompt_pool(
            source_file=preference_config["source_file"],
            output_file=output_file,
            seed=seed,
            pool_size=pool_size,
            exclude_prompt_files=list(preference_config.get("exclude_prompt_files", [])),
        )
        print(f"WROTE preference prompt pool to {written}")
        return

    output_files = preference_config["output_files"]
    train_split_sizes = preference_config["train_split_sizes"]
    eval_split_sizes = preference_config["eval_split_sizes"]
    predictions_file = Path(args.predictions_file or preference_config["rejected_predictions_file"])
    rejected_field = str(preference_config.get("rejected_field", "adapter_response"))
    if args.only_split:
        split_sizes = {**train_split_sizes, **eval_split_sizes}
        if args.only_split not in split_sizes:
            raise KeyError(f"Unknown preference split: {args.only_split}")
        output_file = Path(args.output_file or output_files[args.only_split])
        split_size = int(args.limit or split_sizes[args.only_split])
        if args.dry_run:
            print("PREFERENCE_SINGLE_SPLIT_DRY_RUN_OK")
            print(f"predictions_file: {predictions_file}")
            print(f"output_file: {output_file}")
            print(f"split: {args.only_split}")
            print(f"rows: {split_size}")
            return
        written = build_preference_pairs_from_predictions(
            predictions_file=predictions_file,
            output_file=output_file,
            limit=split_size,
            rejected_field=rejected_field,
        )
        print(f"WROTE {args.only_split} preference rows to {written}")
        return

    if args.dry_run:
        print("PREFERENCE_PAIRS_DRY_RUN_OK")
        print(f"predictions_file: {predictions_file}")
        print(f"rejected_field: {rejected_field}")
        for split_name, size in {**train_split_sizes, **eval_split_sizes}.items():
            print(f"{split_name}: {output_files[split_name]} ({size} rows)")
        return

    written = build_all_preference_splits_from_predictions(
        predictions_file=predictions_file,
        output_files=output_files,
        train_split_sizes=train_split_sizes,
        eval_split_sizes=eval_split_sizes,
        rejected_field=rejected_field,
    )
    for split_name, path in written.items():
        print(f"WROTE {split_name} preference rows to {path}")


if __name__ == "__main__":
    main()
