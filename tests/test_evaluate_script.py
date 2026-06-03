import unittest

import scripts.evaluate as evaluate


class EvaluateScriptTest(unittest.TestCase):
    def test_parse_args_accepts_dry_run_and_limit(self):
        args = evaluate.parse_args(["--dry_run", "--limit", "3"])

        self.assertTrue(args.dry_run)
        self.assertEqual(args.limit, 3)

    def test_parse_args_accepts_skip_flags(self):
        args = evaluate.parse_args(["--skip_generation", "--skip_perplexity"])

        self.assertTrue(args.skip_generation)
        self.assertTrue(args.skip_perplexity)

    def test_parse_args_accepts_generation_length_override(self):
        args = evaluate.parse_args(["--max_new_tokens", "32"])

        self.assertEqual(args.max_new_tokens, 32)

    def test_apply_prediction_limit_limits_reused_prediction_rows(self):
        rows = [
            {"id": "sample_0", "adapter_response": "a"},
            {"id": "sample_1", "adapter_response": "b"},
            {"id": "sample_2", "adapter_response": "c"},
        ]

        limited = evaluate.apply_prediction_limit(rows, limit=2)

        self.assertEqual([row["id"] for row in limited], ["sample_0", "sample_1"])

    def test_rank4_1k_eval_config_uses_isolated_paths(self):
        config = evaluate.load_config("configs/eval_rank4_1k.yaml")

        self.assertEqual(config["model"]["adapter_dir"], "outputs/qwen05b_qlora_1k_r4")
        self.assertEqual(config["outputs"]["predictions_file"], "results/predictions_qwen05b_qlora_1k_r4.jsonl")
        self.assertEqual(config["outputs"]["metrics_file"], "results/metrics_qwen05b_qlora_1k_r4.json")


if __name__ == "__main__":
    unittest.main()
