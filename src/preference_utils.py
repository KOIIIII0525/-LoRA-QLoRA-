"""Utilities for building lightweight DPO preference datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from src.data_utils import has_user_and_assistant, write_jsonl


def extract_prompt_reference(sample: list[dict[str, Any]]) -> tuple[str, str]:
    """Extract the last user prompt and its following assistant answer."""
    prompt = ""
    reference = ""
    for index, message in enumerate(sample):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        prompt = content.strip()
        reference = ""
        for next_message in sample[index + 1 :]:
            if not isinstance(next_message, dict) or next_message.get("role") != "assistant":
                continue
            next_content = next_message.get("content", "")
            reference = next_content.strip() if isinstance(next_content, str) else ""
            break
    return prompt, reference


def validate_preference_row(row: dict[str, Any]) -> bool:
    """Return True when a row is usable for DPO-style preference training."""
    prompt = row.get("prompt", "")
    chosen = row.get("chosen", "")
    rejected = row.get("rejected", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    if not isinstance(chosen, str) or not chosen.strip():
        return False
    if not isinstance(rejected, str) or not rejected.strip():
        return False
    return chosen.strip() != rejected.strip()


def build_preference_row(
    sample_id: str,
    prompt: str,
    chosen: str,
    rejected: str,
    source: str = "reference_vs_generated",
) -> dict[str, str]:
    """Build a stable chosen/rejected preference record."""
    return {
        "id": sample_id,
        "prompt": prompt.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "source": source,
    }


def build_preference_splits_from_rows(
    rows: list[Any],
    rejected_by_prompt: dict[str, str],
    seed: int,
    split_sizes: dict[str, int],
    output_paths: dict[str, str | Path],
) -> dict[str, Path]:
    """Build deterministic, non-overlapping preference splits from chat rows."""
    candidates: list[dict[str, str]] = []
    for source_index, row in enumerate(rows):
        if not has_user_and_assistant(row):
            continue
        prompt, reference = extract_prompt_reference(row)
        rejected = rejected_by_prompt.get(prompt, "")
        preference_row = build_preference_row(
            sample_id=f"pref_{source_index}",
            prompt=prompt,
            chosen=reference,
            rejected=rejected,
        )
        if validate_preference_row(preference_row):
            candidates.append(preference_row)

    required = sum(split_sizes.values())
    if len(candidates) < required:
        raise ValueError(f"Need at least {required} valid preference rows, got {len(candidates)}")

    rng = random.Random(seed)
    rng.shuffle(candidates)

    written: dict[str, Path] = {}
    cursor = 0
    for split_name, size in split_sizes.items():
        path = Path(output_paths[split_name])
        split_rows = candidates[cursor : cursor + size]
        cursor += size
        write_jsonl(path, split_rows)
        written[split_name] = path

    return written
