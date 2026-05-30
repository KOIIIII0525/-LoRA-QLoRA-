"""Prepare small deterministic SFT splits from processed BELLE chat data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import build_sft_splits, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare lightweight SFT data splits.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_qwen_0.5b.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config["project"].get("seed", 42))
    data_config = config["data"]

    output_paths = {
        "train_100": data_config["train_100"],
        "train_1k": data_config["train_1k"],
        "train_3k": data_config["train_3k"],
        "valid": data_config["valid_300"],
        "test": data_config["test_100"],
        "manual_prompts": data_config["manual_prompts_30"],
    }

    written = build_sft_splits(
        source_file=data_config["source_file"],
        output_dir=data_config["output_dir"],
        seed=seed,
        train_sizes={"train_100": 100, "train_1k": 1000, "train_3k": 3000},
        valid_size=300,
        test_size=100,
        manual_prompt_size=30,
        output_paths=output_paths,
    )

    print("Prepared SFT data splits:")
    for name, path in written.items():
        print(f"- {name}: {path} ({len(load_jsonl(path))} rows)")


if __name__ == "__main__":
    main()
