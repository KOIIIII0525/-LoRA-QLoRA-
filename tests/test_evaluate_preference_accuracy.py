import unittest
from pathlib import Path

import scripts.evaluate_preference_accuracy as evaluate_preference_accuracy


class EvaluatePreferenceAccuracyScriptTest(unittest.TestCase):
    def test_parse_args_accepts_dry_run_limit_and_output_file(self):
        args = evaluate_preference_accuracy.parse_args(
            ["--dry_run", "--limit", "5", "--output_file", "results/pref_acc.json"]
        )

        self.assertTrue(args.dry_run)
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.output_file, "results/pref_acc.json")

    def test_resolve_preference_eval_inputs_uses_test_split_and_adapters(self):
        config = evaluate_preference_accuracy.load_config("configs/dpo_qwen_0.5b.yaml")

        resolved = evaluate_preference_accuracy.resolve_preference_eval_inputs(config, None, None, None, None)

        self.assertEqual(resolved["preference_file"], Path("data/processed/preference_test_100.jsonl"))
        self.assertEqual(resolved["sft_adapter_dir"], Path("outputs/qwen05b_qlora_3k"))
        self.assertEqual(resolved["dpo_adapter_dir"], Path("outputs/qwen05b_dpo_300"))
        self.assertEqual(resolved["output_file"], Path("results/preference_accuracy_sft_dpo_qwen05b.json"))


if __name__ == "__main__":
    unittest.main()
