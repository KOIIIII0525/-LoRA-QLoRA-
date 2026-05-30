"""Pure helpers for inference and evaluation scripts."""

from __future__ import annotations

import math
from typing import Any


def normalize_prompt_record(record: Any, index: int) -> dict[str, str]:
    """Normalize manual-prompt dicts or chat-message lists to prompt/reference rows."""
    if isinstance(record, dict):
        prompt = record.get("prompt", "")
        reference = record.get("reference", "")
        sample_id = record.get("id", f"sample_{index}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing prompt in record {index}")
        return {
            "id": str(sample_id),
            "prompt": prompt,
            "reference": reference if isinstance(reference, str) else "",
        }

    if isinstance(record, list):
        prompt = ""
        reference = ""
        for message_index, message in enumerate(record):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            prompt = content
            reference = ""
            for next_message in record[message_index + 1 :]:
                if isinstance(next_message, dict) and next_message.get("role") == "assistant":
                    next_content = next_message.get("content", "")
                    reference = next_content if isinstance(next_content, str) else ""
                    break

        if not prompt:
            raise ValueError(f"Missing user prompt in record {index}")
        return {
            "id": f"sample_{index}",
            "prompt": prompt,
            "reference": reference,
        }

    raise ValueError(f"Unsupported record type at index {index}: {type(record).__name__}")


def build_prediction_record(
    sample_id: str,
    prompt: str,
    reference: str,
    base_response: str | None = None,
    adapter_response: str | None = None,
) -> dict[str, str]:
    """Build a stable JSONL prediction row."""
    return {
        "id": sample_id,
        "prompt": prompt,
        "reference": reference,
        "base_response": base_response or "",
        "adapter_response": adapter_response or "",
    }


def safe_perplexity(loss: float) -> float:
    """Convert loss to perplexity without raising on overflow."""
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


def summarize_losses(losses: list[float | None]) -> dict[str, float | int | None]:
    """Summarize non-null loss values."""
    values = [float(loss) for loss in losses if loss is not None]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
