"""Online prompt bucketing preserves coverage, replay, and epoch identity."""

import copy
import os
import random
import unittest
from unittest.mock import patch

from specforge.launch import (
    _epoch_online_prompts,
    _epoch_prompt_indices,
    _iter_epoch_online_prompt_batches,
    _length_bucket_size,
    _prompt_length,
)

BUCKET_ENV = "SPECFORGE_LENGTH_BUCKET"


def prompts_for(lengths):
    return [
        {
            "task_id": f"p{index}",
            "payload": {"input_ids": [1] * length, "loss_mask": [1] * length},
            "metadata": {"source": "test"},
        }
        for index, length in enumerate(lengths)
    ]


class PromptLengthBucketingTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {BUCKET_ENV: "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_default_retains_original_random_order(self):
        os.environ.pop(BUCKET_ENV)
        prompts = prompts_for(range(1, 20))
        expected = list(range(len(prompts)))
        random.Random(43).shuffle(expected)
        self.assertEqual(_length_bucket_size(), 0)
        self.assertEqual(_epoch_prompt_indices(prompts, 1, seed=42), expected)

    def test_disabled_and_invalid_settings_retain_original_shuffle(self):
        prompts = prompts_for(range(1, 20))
        expected = list(range(len(prompts)))
        random.Random(7).shuffle(expected)
        for value in ("", " ", "0", "1", "-10", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {BUCKET_ENV: value}):
                self.assertEqual(_epoch_prompt_indices(prompts, 0, seed=7), expected)

    def test_length_uses_exact_tokens_or_text_proxy(self):
        cases = [
            ({"payload": {"input_ids": [1, 2]}, "prompt": "long text"}, 2),
            ({"prompt": "hello"}, 5),
            ({"messages": [{"content": "hello"}, {"content": "world"}]}, 10),
            (
                {"messages": [{"content": [{"text": "hi"}, "abc", {"image": "x"}]}]},
                5,
            ),
            ({"conversations": [{"content": "hello"}]}, 5),
            ({"payload": {"input_ids": None}}, 0),
            ({"payload": {"input_ids": 1}}, 0),
            (None, 0),
        ]
        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(_prompt_length(prompt), expected)

    def test_unknown_lengths_fall_back_to_seeded_shuffle(self):
        os.environ[BUCKET_ENV] = "4"
        prompts = [{"task_id": f"p{i}"} for i in range(12)]
        expected = list(range(len(prompts)))
        random.Random(8).shuffle(expected)
        self.assertEqual(_epoch_prompt_indices(prompts, 1, seed=7), expected)
        self.assertEqual(_epoch_prompt_indices([], 0), [])

    def test_each_block_keeps_every_sample_and_sorts_by_length(self):
        os.environ[BUCKET_ENV] = "4"
        lengths = [9, 1, 7, 3, 8, 2, 6, 4, 11, 5, 10, 12]
        prompts = prompts_for(lengths)
        original = copy.deepcopy(prompts)
        order = _epoch_prompt_indices(prompts, 0, seed=42)
        self.assertCountEqual(order, range(len(prompts)))
        for start in range(0, len(order), 4):
            block = order[start : start + 4]
            self.assertEqual(len({index // 4 for index in block}), 1)
            self.assertEqual(
                [lengths[index] for index in block],
                sorted(lengths[index] for index in block),
            )
        self.assertEqual(prompts, original)

    def test_plan_rebuilds_and_block_order_changes_with_seed_and_epoch(self):
        os.environ[BUCKET_ENV] = "4"
        prompts = prompts_for(range(32, 0, -1))
        order = _epoch_prompt_indices(prompts, 0, seed=0)
        self.assertEqual(order, _epoch_prompt_indices(prompts, 0, seed=0))
        self.assertNotEqual(order, _epoch_prompt_indices(prompts, 0, seed=1))
        self.assertNotEqual(order, _epoch_prompt_indices(prompts, 1, seed=0))

    def test_single_block_retains_sorted_order_across_epochs(self):
        os.environ[BUCKET_ENV] = "2048"
        prompts = prompts_for([7, 1, 3])
        self.assertEqual(_epoch_prompt_indices(prompts, 0, seed=42), [1, 2, 0])
        self.assertEqual(_epoch_prompt_indices(prompts, 1, seed=42), [1, 2, 0])

    def test_chunked_plan_preserves_partial_block_and_unique_epoch_ids(self):
        os.environ[BUCKET_ENV] = "4"
        prompts = prompts_for(range(17, 0, -1))
        original = copy.deepcopy(prompts)
        all_ids = []
        for epoch in range(3):
            batches = list(
                _iter_epoch_online_prompt_batches(
                    prompts, epoch, 3, seed=42, batch_size=5
                )
            )
            plan = [item for batch in batches for item in batch]
            self.assertEqual(plan, _epoch_online_prompts(prompts, epoch, 3, seed=42))
            self.assertEqual([len(batch) for batch in batches], [5, 5, 5, 2])
            self.assertCountEqual(
                [item["metadata"]["prompt_index"] for item in plan], range(17)
            )
            for item in plan:
                index = item["metadata"]["prompt_index"]
                self.assertEqual(item["metadata"]["epoch"], epoch)
                self.assertEqual(item["metadata"]["base_task_id"], f"p{index}")
                self.assertEqual(item["payload"], prompts[index]["payload"])
                all_ids.append(item["task_id"])
        self.assertEqual(len(set(all_ids)), 51)
        self.assertEqual(prompts, original)

    def test_bucketing_reduces_synthetic_padding_without_dropping_tokens(self):
        lengths = [1, 100] * 32
        prompts = prompts_for(lengths)
        random_order = _epoch_prompt_indices(prompts, 0, seed=42)
        os.environ[BUCKET_ENV] = "8"
        bucket_order = _epoch_prompt_indices(prompts, 0, seed=42)

        def padded_tokens(order):
            return sum(
                4 * max(lengths[index] for index in order[start : start + 4])
                for start in range(0, len(order), 4)
            )

        self.assertCountEqual(bucket_order, random_order)
        self.assertEqual(sum(lengths[index] for index in bucket_order), sum(lengths))
        self.assertLess(padded_tokens(bucket_order), padded_tokens(random_order))


if __name__ == "__main__":
    unittest.main()
