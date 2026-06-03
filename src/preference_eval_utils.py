"""Pure helpers for chosen/rejected preference accuracy evaluation."""

from __future__ import annotations

from typing import Any


def preferred_label(chosen_logprob: float, rejected_logprob: float, tie_epsilon: float = 1e-12) -> str:
    """Return which answer has the higher score."""
    margin = float(chosen_logprob) - float(rejected_logprob)
    if abs(margin) <= tie_epsilon:
        return "tie"
    return "chosen" if margin > 0 else "rejected"


def build_preference_score_record(
    sample_id: str,
    prompt: str,
    chosen: str,
    rejected: str,
    chosen_logprob: float,
    rejected_logprob: float,
) -> dict[str, Any]:
    """Build a stable score row for one model on one preference sample."""
    margin = float(chosen_logprob) - float(rejected_logprob)
    return {
        "id": sample_id,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_logprob": float(chosen_logprob),
        "rejected_logprob": float(rejected_logprob),
        "margin": margin,
        "preferred": preferred_label(float(chosen_logprob), float(rejected_logprob)),
    }


def summarize_preference_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize how often chosen beats rejected."""
    count = len(records)
    chosen_preferred = sum(1 for row in records if row.get("preferred") == "chosen")
    rejected_preferred = sum(1 for row in records if row.get("preferred") == "rejected")
    ties = sum(1 for row in records if row.get("preferred") == "tie")

    if count == 0:
        return {
            "count": 0,
            "chosen_preferred": 0,
            "rejected_preferred": 0,
            "ties": 0,
            "accuracy": None,
            "mean_chosen_logprob": None,
            "mean_rejected_logprob": None,
            "mean_margin": None,
        }

    chosen_scores = [float(row["chosen_logprob"]) for row in records]
    rejected_scores = [float(row["rejected_logprob"]) for row in records]
    margins = [float(row["margin"]) for row in records]
    return {
        "count": count,
        "chosen_preferred": chosen_preferred,
        "rejected_preferred": rejected_preferred,
        "ties": ties,
        "accuracy": chosen_preferred / count,
        "mean_chosen_logprob": sum(chosen_scores) / count,
        "mean_rejected_logprob": sum(rejected_scores) / count,
        "mean_margin": sum(margins) / count,
    }
