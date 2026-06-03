import unittest

import scripts.infer as infer


class InferScriptTest(unittest.TestCase):
    def test_parse_args_accepts_prompt_and_dry_run(self):
        args = infer.parse_args(["--prompt", "请解释 LoRA", "--mode", "both", "--dry_run"])

        self.assertEqual(args.prompt, "请解释 LoRA")
        self.assertEqual(args.mode, "both")
        self.assertTrue(args.dry_run)

    def test_parse_args_accepts_batch_limit(self):
        args = infer.parse_args(["--input_file", "data/processed/manual_prompts_30.jsonl", "--limit", "2"])

        self.assertEqual(args.input_file, "data/processed/manual_prompts_30.jsonl")
        self.assertEqual(args.limit, 2)

    def test_resolve_prompt_rows_applies_offset_before_limit(self):
        class Args:
            prompt = None
            input_file = "tests/.tmp/infer_offset_prompts.jsonl"
            offset = 2
            limit = 2

        from src.data_utils import write_jsonl

        write_jsonl(
            Args.input_file,
            [
                {"id": "sample_0", "prompt": "问题0", "reference": "回答0"},
                {"id": "sample_1", "prompt": "问题1", "reference": "回答1"},
                {"id": "sample_2", "prompt": "问题2", "reference": "回答2"},
                {"id": "sample_3", "prompt": "问题3", "reference": "回答3"},
            ],
        )

        rows = infer.resolve_prompt_rows({"data": {"manual_prompts_file": Args.input_file}}, Args)

        self.assertEqual([row["id"] for row in rows], ["sample_2", "sample_3"])


if __name__ == "__main__":
    unittest.main()
