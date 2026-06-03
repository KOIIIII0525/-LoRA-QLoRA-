import json
import unittest
import uuid
from pathlib import Path

from src.data_utils import load_jsonl
from src.preference_utils import (
    build_preference_splits_from_rows,
    extract_prompt_reference,
    validate_preference_row,
)


class PreferenceUtilsTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path("tests/.tmp") / f"{self._testMethodName}_{uuid.uuid4().hex}"
        self.tmp_root.mkdir(parents=True)

    def test_extract_prompt_reference_uses_last_user_and_following_assistant(self):
        sample = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "第二轮问题"},
            {"role": "assistant", "content": "第二轮回答"},
        ]

        prompt, reference = extract_prompt_reference(sample)

        self.assertEqual(prompt, "第二轮问题")
        self.assertEqual(reference, "第二轮回答")

    def test_validate_preference_row_rejects_empty_or_identical_outputs(self):
        valid = {
            "id": "pref_0",
            "prompt": "解释 LoRA",
            "chosen": "LoRA 是低秩适配方法。",
            "rejected": "LoRA 是一种无线通信技术。",
            "source": "reference_vs_generated",
        }

        self.assertTrue(validate_preference_row(valid))
        self.assertFalse(validate_preference_row({**valid, "chosen": ""}))
        self.assertFalse(validate_preference_row({**valid, "rejected": ""}))
        self.assertFalse(validate_preference_row({**valid, "rejected": valid["chosen"]}))

    def test_build_preference_splits_are_deterministic_and_non_overlapping(self):
        rows = []
        rejected_by_prompt = {}
        for idx in range(12):
            prompt = f"问题{idx}"
            rows.append(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"参考回答{idx}"},
                ]
            )
            rejected_by_prompt[prompt] = f"模型回答{idx}"

        output_paths = {
            "train_3": self.tmp_root / "preference_train_3.jsonl",
            "valid_2": self.tmp_root / "preference_valid_2.jsonl",
            "test_2": self.tmp_root / "preference_test_2.jsonl",
        }

        first = build_preference_splits_from_rows(
            rows=rows,
            rejected_by_prompt=rejected_by_prompt,
            seed=7,
            split_sizes={"train_3": 3, "valid_2": 2, "test_2": 2},
            output_paths=output_paths,
        )
        second = build_preference_splits_from_rows(
            rows=rows,
            rejected_by_prompt=rejected_by_prompt,
            seed=7,
            split_sizes={"train_3": 3, "valid_2": 2, "test_2": 2},
            output_paths=output_paths,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(load_jsonl(first["train_3"])), 3)
        self.assertEqual(len(load_jsonl(first["valid_2"])), 2)
        self.assertEqual(len(load_jsonl(first["test_2"])), 2)

        split_prompts = {
            name: {row["prompt"] for row in load_jsonl(path)}
            for name, path in first.items()
        }
        self.assertTrue(split_prompts["train_3"].isdisjoint(split_prompts["valid_2"]))
        self.assertTrue(split_prompts["train_3"].isdisjoint(split_prompts["test_2"]))
        self.assertTrue(split_prompts["valid_2"].isdisjoint(split_prompts["test_2"]))

        train_row = load_jsonl(first["train_3"])[0]
        self.assertEqual(set(train_row), {"id", "prompt", "chosen", "rejected", "source"})
        self.assertNotEqual(train_row["chosen"], train_row["rejected"])


if __name__ == "__main__":
    unittest.main()
