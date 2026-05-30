# Stage 6 GitHub Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the completed local QLoRA experiment into a clean, credible GitHub-ready project package without committing model weights, raw data, training outputs, or large local result files.

**Architecture:** Keep heavy artifacts local and ignored; commit only source code, configs, tests, docs, small placeholders, and curated result summaries. The README should be the entry point, while `docs/` carries reproducibility details, experiment results, error logs, and sample outputs.

**Tech Stack:** Git, Markdown, Python unittest, existing Python scripts, existing QLoRA/PEFT project structure.

---

## File Structure

Files to modify or create during Stage 6:

- Modify: `README.md`  
  Main GitHub-facing project page: positioning, quick start, experiment table, evaluation summary, limitations, repository structure.
- Modify: `docs/progress.md`  
  Record Stage 6 progress and verification checkpoints.
- Modify: `docs/experiment_results.md`  
  Keep detailed training/evaluation results and conservative interpretation.
- Modify: `docs/setup.md`  
  Ensure environment setup and model download instructions are enough for reproduction.
- Modify: `docs/result_samples.md`  
  Keep a small commit-safe result excerpt because `results/*.json/jsonl` stays ignored.
- Modify: `.gitignore`  
  Ensure large data, model weights, outputs, caches, local result files, and temp files are excluded.
- Optional create: `assets/demo_infer.png` or `assets/demo_infer.txt`  
  Add only small demo material. Avoid committing generated model files or large screenshots.

---

### Task 1: Final Git Ignore Audit

**Files:**
- Modify: `.gitignore`
- Modify: `docs/progress.md`

- [x] **Step 1: Run ignored status audit**

Run:

```powershell
git status --ignored --short -uall
```

Expected:

- `models/`, `outputs/`, `base_model_215M/`, raw BELLE/SeqMonkey files, `data/processed/`, `results/*.json`, and `results/*.jsonl` appear as ignored.
- Source code, configs, tests, docs, `.gitignore`, `.gitkeep` placeholders, and small curated docs appear as untracked or tracked.

- [x] **Step 2: Patch `.gitignore` only if unsafe files are visible**

If unexpected local/cache files appear as untracked, add narrow ignore rules. Do not ignore source directories broadly.

- [x] **Step 3: Re-run ignored status audit**

Run:

```powershell
git status --ignored --short -uall
```

Expected: no large data/model/output artifact appears as untracked.

---

### Task 2: README GitHub Polish

**Files:**
- Modify: `README.md`
- Cross-check: `docs/experiment_results.md`
- Cross-check: `docs/result_samples.md`

- [x] **Step 1: Review README for GitHub reader flow**

Check that the top sections answer:

- What is this project?
- What hardware and model does it use?
- What has already been run?
- What commands reproduce data preparation, smoke test, main training, inference, and evaluation?
- What are the results and limitations?

- [x] **Step 2: Tighten commands**

Ensure commands are explicit:

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_3k --output_dir outputs\qwen05b_qlora_3k
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\infer.py --config configs\eval.yaml --prompt "请解释什么是 LoRA" --mode adapter
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\evaluate.py --config configs\eval.yaml
```

- [x] **Step 3: Keep claims conservative**

Expected wording:

- Use “观察到部分改善”.
- Avoid “显著提升”, “工业级”, “SOTA”.
- Mention failure cases and limitations.

---

### Task 3: Result Summary and Demo Material

**Files:**
- Modify: `docs/result_samples.md`
- Optional create: `assets/demo_infer.txt` or small screenshot asset

- [x] **Step 1: Verify result summary mirrors local metrics**

Run:

```powershell
Get-Content -Raw results\metrics_qwen05b_qlora_3k.json
```

Expected:

- examples: 100
- ROUGE-L: about 0.2539
- eval loss: about 1.8221
- perplexity: about 6.1851

- [x] **Step 2: Keep only small curated samples**

Use `docs/result_samples.md` for 2-3 representative samples:

- One structured-output improvement.
- One extraction/task-following improvement.
- One failure case.

- [x] **Step 3: Optional demo screenshot/text**

No extra asset was added in this pass. `docs/result_samples.md` is enough for the first GitHub showcase, and full `results/` JSONL / JSON files remain ignored.

---

### Task 4: Reproducibility Check

**Files:**
- Modify: `docs/setup.md` if gaps are found
- Modify: `README.md` if command gaps are found

- [x] **Step 1: Run lightweight checks**

Run:

```powershell
python -B -m unittest discover -s tests -v
python -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --dry_run
python -B scripts\infer.py --config configs\eval.yaml --input_file data\processed\manual_prompts_30.jsonl --limit 2 --mode both --dry_run
python -B scripts\evaluate.py --config configs\eval.yaml --limit 2 --dry_run
```

Expected:

- Unit tests pass.
- Training dry-run uses `train_3k` by default.
- Inference and evaluation dry-run print expected paths and record counts.

- [x] **Step 2: Run heavy checks only when needed**

Use `pytorch_env` for commands that import training/evaluation dependencies:

```powershell
D:\anaconda3\envs\pytorch_env\python.exe -B scripts\train_lora.py --config configs\lora_qwen_0.5b.yaml --train_split train_100 --check_env
```

Expected:

- CUDA available.
- QLoRA dependencies import correctly.

Result on 2026-05-31:

- `ENV_CHECK_OK`
- `torch: 2.10.0+cu126`
- `cuda_available: True`
- `device: NVIDIA GeForce RTX 3060 Laptop GPU`

---

### Task 5: Commit Boundary Review

**Files:**
- No code changes expected unless audit finds problems.

- [x] **Step 1: Inspect untracked files**

Run:

```powershell
git status --short
```

Expected commit candidates:

- `.gitignore`
- `AGENTS.md`
- `README.md`
- `requirements.txt`
- `configs/`
- `docs/`
- `scripts/`
- `src/`
- `tests/`
- `.gitkeep` placeholders
- optionally `tiny_llm_legacy/` and `tokenizer_k/` if you want to preserve old learning code

- [x] **Step 2: Decide whether to keep legacy code in GitHub repo**

Decision point:

- Keep `tiny_llm_legacy/` if you want to show project evolution from Tiny LLM to QLoRA.
- Exclude or move it if you want a cleaner focused LoRA/QLoRA project.

Decision on 2026-05-31:

- Do not include `tiny_llm_legacy/` in the first GitHub commit. It is useful local history, but it is not part of the Qwen2.5 QLoRA training / inference / evaluation path and would distract from the minimal closed loop.
- Do not include `tokenizer_k/` in the first GitHub commit. It is an old custom tokenizer artifact and is unrelated to the current local Qwen tokenizer.

- [ ] **Step 3: Stage intentionally**

Do not use broad staging until the ignored audit is clean.

Suggested:

```powershell
git add .gitignore AGENTS.md README.md requirements.txt configs docs scripts src tests assets/.gitkeep data/samples/.gitkeep results/.gitkeep
git status --short
```

Expected: no model weights, raw datasets, `outputs/`, `data/processed/`, or `results/*.json/jsonl` staged.

---

## Stage 6 Completion Criteria

- [x] `git status --ignored --short -uall` confirms large local artifacts are ignored.
- [x] README has correct main training, inference, and evaluation commands.
- [x] `docs/result_samples.md` provides commit-safe evidence of metrics and sample outputs.
- [x] Unit tests pass.
- [x] Progress docs identify Stage 6 status and remaining optional polish.
- [x] Commit candidates are reviewed before any `git add .` or first commit.
