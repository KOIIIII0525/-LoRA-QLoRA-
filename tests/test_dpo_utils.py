import unittest
from pathlib import Path

from src.dpo_utils import (
    build_dpo_arguments_kwargs,
    build_dpo_dry_run_report,
    load_dpo_config,
    resolve_dpo_inputs,
    validate_dpo_inputs,
)


class DpoUtilsTest(unittest.TestCase):
    def test_load_dpo_config_reads_yaml(self):
        config = load_dpo_config("configs/dpo_qwen_0.5b.yaml")

        self.assertEqual(config["model"]["base_model_name_or_path"], "models/qwen2.5-0.5b-instruct")
        self.assertEqual(config["model"]["sft_adapter_dir"], "outputs/qwen05b_qlora_3k")
        self.assertEqual(config["dpo"]["beta"], 0.1)

    def test_resolve_dpo_inputs_accepts_smoke_split(self):
        config = load_dpo_config("configs/dpo_qwen_0.5b.yaml")

        resolved = resolve_dpo_inputs(config, train_split="preference_train_50", output_dir=None)

        self.assertEqual(resolved["train_file"], Path("data/processed/preference_train_50.jsonl"))
        self.assertEqual(resolved["eval_file"], Path("data/processed/preference_valid_100.jsonl"))
        self.assertEqual(resolved["output_dir"], Path("outputs/qwen05b_dpo_50"))
        self.assertEqual(resolved["sft_adapter_dir"], Path("outputs/qwen05b_qlora_3k"))

    def test_validate_dpo_inputs_counts_rows_and_checks_schema(self):
        config = load_dpo_config("configs/dpo_qwen_0.5b.yaml")
        resolved = resolve_dpo_inputs(config, train_split="preference_train_50", output_dir=None)

        summary = validate_dpo_inputs(config, resolved)

        self.assertEqual(summary["train_rows"], 50)
        self.assertEqual(summary["max_length"], 512)
        self.assertEqual(summary["max_prompt_length"], 256)

    def test_build_dpo_dry_run_report_contains_key_settings(self):
        config = load_dpo_config("configs/dpo_qwen_0.5b.yaml")
        resolved = resolve_dpo_inputs(config, train_split="preference_train_50", output_dir=None)
        summary = validate_dpo_inputs(config, resolved)

        report = build_dpo_dry_run_report(config, resolved, summary)

        self.assertIn("DPO training dry run", report)
        self.assertIn("preference_train_50.jsonl", report)
        self.assertIn("outputs\\qwen05b_dpo_50", report.replace("/", "\\"))
        self.assertIn("beta: 0.1", report)

    def test_build_dpo_arguments_disables_eval_when_eval_file_is_missing(self):
        config = load_dpo_config("configs/dpo_qwen_0.5b.yaml")

        class NewDPOConfig:
            def __init__(self, output_dir, eval_strategy=None, save_strategy=None, bf16=None, fp16=False):
                pass

        kwargs = build_dpo_arguments_kwargs(
            config=config,
            output_dir=Path("outputs/qwen05b_dpo_50"),
            dpo_config_cls=NewDPOConfig,
            has_eval_dataset=False,
        )

        self.assertEqual(kwargs["eval_strategy"], "no")
        self.assertFalse(kwargs["bf16"])
        self.assertFalse(kwargs["fp16"])


if __name__ == "__main__":
    unittest.main()
