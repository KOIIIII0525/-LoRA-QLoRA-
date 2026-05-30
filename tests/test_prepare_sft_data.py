import json
import unittest
import uuid
from pathlib import Path

from src.data_utils import (
    build_sft_splits,
    has_user_and_assistant,
    load_jsonl,
    write_jsonl,
)


class PrepareSftDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path("tests/.tmp") / f"{self._testMethodName}_{uuid.uuid4().hex}"
        self.tmp_root.mkdir(parents=True)

    def tearDown(self):
        # Keep test artifacts under tests/.tmp; the directory is ignored by Git.
        # This avoids Windows file-lock cleanup errors in constrained sandboxes.
        pass

    def test_has_user_and_assistant_accepts_valid_chat(self):
        sample = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
        ]

        self.assertTrue(has_user_and_assistant(sample))

    def test_has_user_and_assistant_rejects_invalid_chat(self):
        self.assertFalse(has_user_and_assistant([{"role": "user", "content": "你好"}]))
        self.assertFalse(has_user_and_assistant([{"role": "assistant", "content": "你好"}]))
        self.assertFalse(has_user_and_assistant({"role": "user", "content": "wrong shape"}))

    def test_load_and_write_jsonl_round_trip(self):
        rows = [
            [{"role": "user", "content": "问题1"}, {"role": "assistant", "content": "回答1"}],
            [{"role": "user", "content": "问题2"}, {"role": "assistant", "content": "回答2"}],
        ]

        path = self.tmp_root / "sample.jsonl"
        write_jsonl(path, rows)

        self.assertEqual(load_jsonl(path), rows)

    def test_build_sft_splits_is_deterministic_and_non_overlapping(self):
        source = self.tmp_root / "source.jsonl"
        rows = []
        for idx in range(20):
            rows.append(
                [
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": f"问题{idx}"},
                    {"role": "assistant", "content": f"回答{idx}"},
                ]
            )
        source.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

        output_dir = self.tmp_root / "processed"
        first = build_sft_splits(
            source_file=source,
            output_dir=output_dir,
            seed=42,
            train_sizes={"train_3": 3, "train_5": 5},
            valid_size=4,
            test_size=3,
            manual_prompt_size=2,
        )
        second = build_sft_splits(
            source_file=source,
            output_dir=output_dir,
            seed=42,
            train_sizes={"train_3": 3, "train_5": 5},
            valid_size=4,
            test_size=3,
            manual_prompt_size=2,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(load_jsonl(first["train_3"])), 3)
        self.assertEqual(len(load_jsonl(first["train_5"])), 5)
        self.assertEqual(len(load_jsonl(first["valid"])), 4)
        self.assertEqual(len(load_jsonl(first["test"])), 3)
        self.assertEqual(len(load_jsonl(first["manual_prompts"])), 2)

        split_contents = {
            name: {json.dumps(item, ensure_ascii=False) for item in load_jsonl(path)}
            for name, path in first.items()
        }

        self.assertTrue(split_contents["train_3"].issubset(split_contents["train_5"]))
        self.assertTrue(split_contents["train_5"].isdisjoint(split_contents["valid"]))
        self.assertTrue(split_contents["train_5"].isdisjoint(split_contents["test"]))
        self.assertTrue(split_contents["valid"].isdisjoint(split_contents["test"]))


if __name__ == "__main__":
    unittest.main()
