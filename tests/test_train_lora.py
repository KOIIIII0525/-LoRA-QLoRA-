import unittest
from pathlib import Path

from src.train_utils import (
    build_trainer_tokenizer_kwargs,
    build_training_arguments_kwargs,
    build_dry_run_report,
    load_training_config,
    resolve_training_inputs,
    validate_training_inputs,
)


class TrainLoraConfigTest(unittest.TestCase):
    def test_load_training_config_reads_yaml(self):
        config = load_training_config("configs/lora_qwen_0.5b.yaml")

        self.assertEqual(config["model"]["base_model_name_or_path"], "models/qwen2.5-0.5b-instruct")
        self.assertEqual(config["data"]["max_seq_len"], 512)
        self.assertEqual(config["lora"]["r"], 8)

    def test_rank4_ablation_config_uses_isolated_output_dir(self):
        config = load_training_config("configs/lora_qwen_0.5b_rank4.yaml")

        self.assertEqual(config["lora"]["r"], 4)
        self.assertEqual(config["lora"]["alpha"], 8)
        self.assertEqual(config["training"]["output_dir"], "outputs/qwen05b_qlora_3k_r4")

    def test_resolve_training_inputs_accepts_explicit_train_100_for_smoke(self):
        config = load_training_config("configs/lora_qwen_0.5b.yaml")

        resolved = resolve_training_inputs(config, train_split="train_100", output_dir=None)

        self.assertEqual(resolved["train_file"], Path("data/processed/train_100.jsonl"))
        self.assertEqual(resolved["eval_file"], Path("data/processed/valid_300.jsonl"))
        self.assertEqual(resolved["output_dir"], Path("outputs/qwen05b_qlora_3k"))

    def test_cli_default_train_split_matches_main_experiment(self):
        import scripts.train_lora as train_lora

        args = train_lora.parse_args([])

        self.assertEqual(args.train_split, "train_3k")

    def test_validate_training_inputs_counts_rows(self):
        config = load_training_config("configs/lora_qwen_0.5b.yaml")
        resolved = resolve_training_inputs(config, train_split="train_100", output_dir=None)

        summary = validate_training_inputs(config, resolved)

        self.assertEqual(summary["train_rows"], 100)
        self.assertEqual(summary["eval_rows"], 300)
        self.assertEqual(summary["max_seq_len"], 512)

    def test_build_dry_run_report_contains_key_settings(self):
        config = load_training_config("configs/lora_qwen_0.5b.yaml")
        resolved = resolve_training_inputs(config, train_split="train_100", output_dir="outputs/smoke")
        summary = validate_training_inputs(config, resolved)

        report = build_dry_run_report(config, resolved, summary)

        self.assertIn("models/qwen2.5-0.5b-instruct", report)
        self.assertIn("train_rows: 100", report)
        self.assertIn("lora_r: 8", report)
        self.assertIn("outputs\\smoke", report.replace("/", "\\"))

    def test_build_training_arguments_kwargs_uses_available_eval_strategy_name(self):
        config = load_training_config("configs/lora_qwen_0.5b.yaml")

        class NewTrainingArguments:
            def __init__(self, output_dir, eval_strategy=None, save_strategy=None):
                pass

        kwargs = build_training_arguments_kwargs(
            config=config,
            output_dir=Path("outputs/smoke"),
            training_arguments_cls=NewTrainingArguments,
        )

        self.assertEqual(kwargs["eval_strategy"], "steps")
        self.assertNotIn("evaluation_strategy", kwargs)

    def test_build_trainer_tokenizer_kwargs_uses_available_processing_name(self):
        class NewTrainer:
            def __init__(self, model=None, processing_class=None):
                pass

        tokenizer = object()
        kwargs = build_trainer_tokenizer_kwargs(NewTrainer, tokenizer)

        self.assertIs(kwargs["processing_class"], tokenizer)
        self.assertNotIn("tokenizer", kwargs)


if __name__ == "__main__":
    unittest.main()
