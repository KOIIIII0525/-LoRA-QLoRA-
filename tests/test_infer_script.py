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


if __name__ == "__main__":
    unittest.main()
