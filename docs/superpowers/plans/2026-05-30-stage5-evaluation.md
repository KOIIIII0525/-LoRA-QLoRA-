# Stage 5 Evaluation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inference and evaluation scripts that complete the Base Model vs QLoRA adapter evaluation loop.

**Architecture:** Keep heavy model loading inside CLI scripts and keep pure data/metric helpers in `src/eval_utils.py` so they can be unit tested without GPU dependencies. `scripts/infer.py` handles single-prompt and batch generation; `scripts/evaluate.py` handles adapter evaluation on `test_100` with ROUGE-L and optional perplexity.

**Tech Stack:** Python, Transformers, PEFT, PyTorch, rouge-score, YAML/JSONL helpers already used by the project.

---

### Task 1: Pure Evaluation Helpers

**Files:**
- Create: `src/eval_utils.py`
- Test: `tests/test_eval_utils.py`

- [ ] Write failing tests for prompt/reference extraction from manual prompt dicts and chat-message samples.
- [ ] Implement `normalize_prompt_record`, `build_prediction_record`, `safe_perplexity`, and `summarize_losses`.
- [ ] Run `python -B -m unittest tests.test_eval_utils -v`.

### Task 2: Inference CLI

**Files:**
- Create: `scripts/infer.py`
- Test: `tests/test_infer_script.py`

- [ ] Write failing import/argument tests for the inference script.
- [ ] Implement CLI with `--prompt`, `--input_file`, `--output_file`, `--mode`, `--limit`, and `--dry_run`.
- [ ] Implement sequential base/adapter loading so both models are not resident at the same time.
- [ ] Run script dry-run against `manual_prompts_30.jsonl`.

### Task 3: Evaluation CLI

**Files:**
- Create: `scripts/evaluate.py`
- Test: `tests/test_evaluate_script.py`

- [ ] Write failing import/argument tests for the evaluation script.
- [ ] Implement CLI with `--config`, `--limit`, `--skip_generation`, `--skip_perplexity`, and `--dry_run`.
- [ ] Compute ROUGE-L from generated predictions and optional eval loss / perplexity.
- [ ] Run script dry-run against `test_100.jsonl`.

### Task 4: Documentation

**Files:**
- Modify: `docs/progress.md`
- Modify: `docs/experiment_results.md`
- Modify: `README.md`

- [ ] Record Stage 5 sub-stage plan.
- [ ] Document dry-run commands and expected outputs.
- [ ] Leave full metric values blank until real generation/evaluation is run.
