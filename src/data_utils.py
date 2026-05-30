"""Utilities for preparing small SFT datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[Any]:
    """Load a UTF-8 JSONL file."""
    rows: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    """Write rows to a UTF-8 JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_user_and_assistant(sample: Any) -> bool:
    """Return True when a sample is a chat list with user and assistant turns."""
    if not isinstance(sample, list):
        return False

    has_user = False
    has_assistant = False
    for message in sample:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True

    return has_user and has_assistant


def extract_manual_prompt(sample: list[dict[str, str]], source_index: int) -> dict[str, Any]:
    """Extract the last user prompt and following assistant answer for manual checks."""
    user_prompt = ""
    reference = ""

    for idx, message in enumerate(sample):
        if message.get("role") != "user":
            continue
        user_prompt = message.get("content", "")
        reference = ""
        for next_message in sample[idx + 1 :]:
            if next_message.get("role") == "assistant":
                reference = next_message.get("content", "")
                break

    return {
        "id": f"manual_{source_index}",
        "prompt": user_prompt,
        "reference": reference,
    }


def build_sft_splits(
    source_file: str | Path,
    output_dir: str | Path,
    seed: int,
    train_sizes: dict[str, int],
    valid_size: int,
    test_size: int,
    manual_prompt_size: int,
    output_paths: dict[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Build deterministic nested train splits plus held-out validation/test sets."""
    output_dir = Path(output_dir)
    output_paths = output_paths or {}

    rows = [row for row in load_jsonl(source_file) if has_user_and_assistant(row)]
    rng = random.Random(seed)
    indexed_rows = list(enumerate(rows))
    rng.shuffle(indexed_rows)

    max_train_size = max(train_sizes.values(), default=0)
    required = max_train_size + valid_size + test_size + manual_prompt_size
    if len(indexed_rows) < required:
        raise ValueError(
            f"Need at least {required} valid rows, got {len(indexed_rows)} from {source_file}"
        )

    train_pool = indexed_rows[:max_train_size]
    valid_pool = indexed_rows[max_train_size : max_train_size + valid_size]
    test_start = max_train_size + valid_size
    test_pool = indexed_rows[test_start : test_start + test_size]
    manual_start = test_start + test_size
    manual_pool = indexed_rows[manual_start : manual_start + manual_prompt_size]

    written: dict[str, Path] = {}
    for name, size in train_sizes.items():
        path = Path(output_paths.get(name, output_dir / f"{name}.jsonl"))
        write_jsonl(path, [row for _, row in train_pool[:size]])
        written[name] = path

    valid_path = Path(output_paths.get("valid", output_dir / "valid.jsonl"))
    test_path = Path(output_paths.get("test", output_dir / "test.jsonl"))
    manual_path = Path(output_paths.get("manual_prompts", output_dir / "manual_prompts.jsonl"))

    write_jsonl(valid_path, [row for _, row in valid_pool])
    write_jsonl(test_path, [row for _, row in test_pool])
    write_jsonl(
        manual_path,
        [extract_manual_prompt(row, source_index) for source_index, row in manual_pool],
    )

    written["valid"] = valid_path
    written["test"] = test_path
    written["manual_prompts"] = manual_path
    return written

