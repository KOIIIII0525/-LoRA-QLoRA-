import unittest

from src.preference_eval_utils import build_preference_score_record, summarize_preference_scores


class PreferenceEvalUtilsTest(unittest.TestCase):
    def test_build_preference_score_record_marks_chosen_rejected_and_tie(self):
        chosen = build_preference_score_record(
            sample_id="pref_1",
            prompt="p",
            chosen="good",
            rejected="bad",
            chosen_logprob=-0.2,
            rejected_logprob=-0.8,
        )
        rejected = build_preference_score_record(
            sample_id="pref_2",
            prompt="p",
            chosen="good",
            rejected="bad",
            chosen_logprob=-1.1,
            rejected_logprob=-0.7,
        )
        tied = build_preference_score_record(
            sample_id="pref_3",
            prompt="p",
            chosen="good",
            rejected="bad",
            chosen_logprob=-0.5,
            rejected_logprob=-0.5,
        )

        self.assertEqual(chosen["preferred"], "chosen")
        self.assertAlmostEqual(chosen["margin"], 0.6)
        self.assertEqual(rejected["preferred"], "rejected")
        self.assertEqual(tied["preferred"], "tie")

    def test_summarize_preference_scores_counts_accuracy_and_margin(self):
        records = [
            {"chosen_logprob": -0.2, "rejected_logprob": -0.8, "preferred": "chosen", "margin": 0.6},
            {"chosen_logprob": -1.1, "rejected_logprob": -0.7, "preferred": "rejected", "margin": -0.4},
            {"chosen_logprob": -0.5, "rejected_logprob": -0.5, "preferred": "tie", "margin": 0.0},
        ]

        summary = summarize_preference_scores(records)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["chosen_preferred"], 1)
        self.assertEqual(summary["rejected_preferred"], 1)
        self.assertEqual(summary["ties"], 1)
        self.assertAlmostEqual(summary["accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["mean_margin"], (0.6 - 0.4 + 0.0) / 3)


if __name__ == "__main__":
    unittest.main()
