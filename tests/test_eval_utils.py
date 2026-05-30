import math
import unittest

from src.eval_utils import (
    build_prediction_record,
    normalize_prompt_record,
    safe_perplexity,
    summarize_losses,
)


class EvalUtilsTest(unittest.TestCase):
    def test_normalize_prompt_record_accepts_manual_prompt_dict(self):
        record = {
            "id": "manual_1",
            "prompt": "请解释什么是 LoRA",
            "reference": "LoRA 是一种参数高效微调方法。",
        }

        normalized = normalize_prompt_record(record, index=0)

        self.assertEqual(normalized["id"], "manual_1")
        self.assertEqual(normalized["prompt"], "请解释什么是 LoRA")
        self.assertEqual(normalized["reference"], "LoRA 是一种参数高效微调方法。")

    def test_normalize_prompt_record_extracts_last_user_and_following_answer(self):
        record = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答"},
        ]

        normalized = normalize_prompt_record(record, index=3)

        self.assertEqual(normalized["id"], "sample_3")
        self.assertEqual(normalized["prompt"], "第二问")
        self.assertEqual(normalized["reference"], "第二答")

    def test_build_prediction_record_keeps_base_and_adapter_outputs(self):
        record = build_prediction_record(
            sample_id="manual_1",
            prompt="问题",
            reference="参考",
            base_response="base",
            adapter_response="adapter",
        )

        self.assertEqual(record["id"], "manual_1")
        self.assertEqual(record["base_response"], "base")
        self.assertEqual(record["adapter_response"], "adapter")

    def test_safe_perplexity_handles_regular_and_large_loss(self):
        self.assertAlmostEqual(safe_perplexity(0.0), 1.0)
        self.assertTrue(math.isinf(safe_perplexity(1000.0)))

    def test_summarize_losses_ignores_none_values(self):
        summary = summarize_losses([1.0, None, 3.0])

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)


if __name__ == "__main__":
    unittest.main()
