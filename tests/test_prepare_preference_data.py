import unittest
import uuid
from pathlib import Path

import scripts.prepare_preference_data as prepare_preference_data
from src.data_utils import load_jsonl, write_jsonl


class PreparePreferenceDataScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path("tests/.tmp") / f"{self._testMethodName}_{uuid.uuid4().hex}"
        self.tmp_root.mkdir(parents=True)

    def test_parse_args_defaults_to_prompt_stage(self):
        args = prepare_preference_data.parse_args([])

        self.assertEqual(args.stage, "prompts")
        self.assertEqual(args.config, "configs/dpo_qwen_0.5b.yaml")

    def test_parse_args_accepts_single_split_pair_overrides(self):
        args = prepare_preference_data.parse_args(
            [
                "--stage",
                "pairs",
                "--only_split",
                "preference_train_50",
                "--predictions_file",
                "results/chunk.jsonl",
                "--output_file",
                "data/processed/preference_train_50.jsonl",
            ]
        )

        self.assertEqual(args.only_split, "preference_train_50")
        self.assertEqual(args.predictions_file, "results/chunk.jsonl")
        self.assertEqual(args.output_file, "data/processed/preference_train_50.jsonl")

    def test_dpo_config_excludes_existing_sft_and_eval_splits(self):
        config = prepare_preference_data.load_config("configs/dpo_qwen_0.5b.yaml")

        excluded = set(config["preference"]["exclude_prompt_files"])

        self.assertIn("data/processed/train_100.jsonl", excluded)
        self.assertIn("data/processed/train_1k.jsonl", excluded)
        self.assertIn("data/processed/train_3k.jsonl", excluded)
        self.assertIn("data/processed/valid_300.jsonl", excluded)
        self.assertIn("data/processed/test_100.jsonl", excluded)
        self.assertIn("data/processed/manual_prompts_30.jsonl", excluded)

    def test_build_preference_prompts_writes_requested_pool(self):
        source_file = self.tmp_root / "source.jsonl"
        rows = []
        for idx in range(8):
            rows.append(
                [
                    {"role": "user", "content": f"问题{idx}"},
                    {"role": "assistant", "content": f"参考回答{idx}"},
                ]
            )
        write_jsonl(source_file, rows)

        output_file = self.tmp_root / "preference_prompts_pool.jsonl"
        written = prepare_preference_data.build_preference_prompt_pool(
            source_file=source_file,
            output_file=output_file,
            seed=11,
            pool_size=5,
            exclude_prompt_files=[],
        )

        prompt_rows = load_jsonl(written)

        self.assertEqual(written, output_file)
        self.assertEqual(len(prompt_rows), 5)
        self.assertEqual(set(prompt_rows[0]), {"id", "prompt", "reference"})

    def test_build_preference_pairs_uses_adapter_response_as_rejected(self):
        predictions_file = self.tmp_root / "predictions.jsonl"
        write_jsonl(
            predictions_file,
            [
                {
                    "id": "pref_prompt_0",
                    "prompt": "解释 LoRA",
                    "reference": "LoRA 是一种低秩适配方法。",
                    "adapter_response": "LoRA 是一种无线通信技术。",
                },
                {
                    "id": "pref_prompt_1",
                    "prompt": "列出两个日期",
                    "reference": "3月1日、4月30日",
                    "adapter_response": "不知道",
                },
            ],
        )
        output_file = self.tmp_root / "preference_train_2.jsonl"

        written = prepare_preference_data.build_preference_pairs_from_predictions(
            predictions_file=predictions_file,
            output_file=output_file,
            limit=2,
            rejected_field="adapter_response",
        )

        rows = load_jsonl(written)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["chosen"], "LoRA 是一种低秩适配方法。")
        self.assertEqual(rows[0]["rejected"], "LoRA 是一种无线通信技术。")
        self.assertEqual(rows[0]["source"], "reference_vs_adapter_response")

    def test_build_all_preference_splits_keeps_smoke_train_nested_in_main_train(self):
        predictions_file = self.tmp_root / "predictions.jsonl"
        prediction_rows = []
        for idx in range(6):
            prediction_rows.append(
                {
                    "id": f"pref_prompt_{idx}",
                    "prompt": f"问题{idx}",
                    "reference": f"参考回答{idx}",
                    "adapter_response": f"模型回答{idx}",
                }
            )
        write_jsonl(predictions_file, prediction_rows)
        output_files = {
            "preference_train_2": self.tmp_root / "preference_train_2.jsonl",
            "preference_train_3": self.tmp_root / "preference_train_3.jsonl",
            "preference_valid_2": self.tmp_root / "preference_valid_2.jsonl",
            "preference_test_1": self.tmp_root / "preference_test_1.jsonl",
        }

        written = prepare_preference_data.build_all_preference_splits_from_predictions(
            predictions_file=predictions_file,
            output_files=output_files,
            train_split_sizes={"preference_train_2": 2, "preference_train_3": 3},
            eval_split_sizes={"preference_valid_2": 2, "preference_test_1": 1},
            rejected_field="adapter_response",
        )

        train_2_prompts = {row["prompt"] for row in load_jsonl(written["preference_train_2"])}
        train_3_prompts = {row["prompt"] for row in load_jsonl(written["preference_train_3"])}
        valid_prompts = {row["prompt"] for row in load_jsonl(written["preference_valid_2"])}
        test_prompts = {row["prompt"] for row in load_jsonl(written["preference_test_1"])}

        self.assertTrue(train_2_prompts.issubset(train_3_prompts))
        self.assertTrue(train_3_prompts.isdisjoint(valid_prompts))
        self.assertTrue(train_3_prompts.isdisjoint(test_prompts))
        self.assertTrue(valid_prompts.isdisjoint(test_prompts))


if __name__ == "__main__":
    unittest.main()
