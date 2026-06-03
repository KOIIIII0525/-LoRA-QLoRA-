import unittest

import scripts.train_dpo as train_dpo


class TrainDpoScriptTest(unittest.TestCase):
    def test_parse_args_defaults_to_smoke_split(self):
        args = train_dpo.parse_args([])

        self.assertEqual(args.config, "configs/dpo_qwen_0.5b.yaml")
        self.assertEqual(args.train_split, "preference_train_50")

    def test_parse_args_accepts_dry_run_and_check_env(self):
        args = train_dpo.parse_args(["--dry_run", "--check_env"])

        self.assertTrue(args.dry_run)
        self.assertTrue(args.check_env)


if __name__ == "__main__":
    unittest.main()
