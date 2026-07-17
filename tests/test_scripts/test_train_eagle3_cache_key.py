import contextlib
import types
import unittest
from unittest import mock

from scripts import train_eagle3


class VocabMappingCacheKeyTest(unittest.TestCase):
    @staticmethod
    def _args():
        return types.SimpleNamespace(
            target_model_path="target/model",
            trust_remote_code=False,
            train_data_path="train.jsonl",
            train_hidden_states_path=None,
            max_length=128,
            chat_template="chat-template",
            cache_dir="/tmp/cache",
            is_vlm=False,
            is_preformatted=False,
            build_dataset_num_proc=1,
            train_only_last_turn=False,
            ttt_length=1,
            attention_backend="sdpa",
            target_batch_size=1,
            dataloader_num_workers=0,
            eval_data_path=None,
            eval_hidden_states_path=None,
        )

    def _keys(self, draft_vocab_size, target_vocab_size):
        config = types.SimpleNamespace(
            draft_vocab_size=draft_vocab_size,
            vocab_size=target_vocab_size,
        )
        with (
            mock.patch.object(train_eagle3, "load_tokenizer", return_value=object()),
            mock.patch.object(
                train_eagle3.Dataset, "from_generator", return_value=object()
            ),
            mock.patch.object(
                train_eagle3, "build_eagle3_dataset", return_value=object()
            ) as build_dataset,
            mock.patch.object(
                train_eagle3,
                "generate_vocab_mapping_file",
                return_value="mapping.pt",
            ) as build_mapping,
            mock.patch.object(
                train_eagle3, "prepare_dp_dataloaders", return_value=object()
            ),
            mock.patch.object(
                train_eagle3,
                "rank_0_priority",
                return_value=contextlib.nullcontext(),
            ),
            mock.patch.object(train_eagle3, "get_dp_group", return_value=None),
        ):
            train_eagle3.build_dataloaders(self._args(), config)
        return (
            build_dataset.call_args.kwargs["cache_key"],
            build_mapping.call_args.kwargs["cache_key"],
        )

    def test_vocab_sizes_only_change_mapping_key(self):
        dataset_key, mapping_key = self._keys(64, 256)
        same_dataset_key, draft_changed_key = self._keys(65, 256)
        target_dataset_key, target_changed_key = self._keys(64, 257)

        self.assertEqual(dataset_key, same_dataset_key)
        self.assertEqual(dataset_key, target_dataset_key)
        self.assertNotEqual(mapping_key, draft_changed_key)
        self.assertNotEqual(mapping_key, target_changed_key)

    def test_identical_inputs_are_stable(self):
        self.assertEqual(self._keys(64, 256), self._keys(64, 256))


if __name__ == "__main__":
    unittest.main(verbosity=2)
