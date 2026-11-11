"""Vocabulary mappings are explicit tensors with safe installation and HF restore."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

from specforge.modeling.draft.models.vocabulary import DraftVocabulary
from specforge.modeling.draft.pretrained import DraftPreTrainedModel
from tests.test_modeling.test_draft_pretrained import model_config


class DFlashDraftModel(DraftPreTrainedModel):
    """Test owner for the shared HF initialization/restoration path."""

    def __init__(self, config):
        super().__init__(config)
        self.projection = torch.nn.Linear(4, 4)
        self.vocabulary = DraftVocabulary(vocab_size=8, draft_vocab_size=4)
        self.post_init()


def install(vocabulary):
    vocabulary.load_mapping(
        t2d=torch.tensor([True, False, True, False, False, True, False, True]),
        d2t=torch.tensor([0, 1, 3, 4]),
    )


class DraftVocabularyTest(unittest.TestCase):
    def test_identity_and_reduced_mapping(self):
        identity = DraftVocabulary(vocab_size=8, draft_vocab_size=8)
        ids = torch.tensor([[0, 7], [2, 4]], dtype=torch.int32)
        torch.testing.assert_close(identity.to_target(ids), ids.long(), rtol=0, atol=0)
        reduced = DraftVocabulary(vocab_size=8, draft_vocab_size=4)
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            reduced.to_target(torch.tensor([0]))
        buffers = [id(t) for t in reduced.buffers()]
        install(reduced)
        self.assertEqual(buffers, [id(t) for t in reduced.buffers()])
        torch.testing.assert_close(
            reduced.to_target(torch.tensor([[3, 0], [1, 2]])),
            torch.tensor([[7, 0], [2, 5]]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(list(reduced.parameters()), [])

    def test_invalid_mapping_preserves_installed_state(self):
        vocabulary = DraftVocabulary(vocab_size=8, draft_vocab_size=4)
        install(vocabulary)
        state = {n: t.clone() for n, t in vocabulary.state_dict().items()}
        for t2d, d2t in (
            (vocabulary.t2d.long(), vocabulary.d2t),
            (vocabulary.t2d, vocabulary.d2t.int()),
            (torch.ones(8, dtype=torch.bool), vocabulary.d2t),
            (vocabulary.t2d, torch.tensor([0, 3, 1, 4])),
            (vocabulary.t2d, torch.tensor([0, 1, 3, 99])),
        ):
            with self.assertRaises(ValueError):
                vocabulary.load_mapping(t2d=t2d, d2t=d2t)
            torch.testing.assert_close(vocabulary.state_dict(), state, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            vocabulary.to_target(torch.tensor([-1]))
        with self.assertRaises(TypeError):
            vocabulary.to_target(torch.tensor([0.0]))

    def test_hf_restore_and_meta_placement(self):
        owner = DFlashDraftModel(model_config())
        install(owner.vocabulary)
        with TemporaryDirectory() as path:
            owner.save_pretrained(path)
            restored = DFlashDraftModel.from_pretrained(path)
        torch.testing.assert_close(
            restored.vocabulary.state_dict(),
            owner.vocabulary.state_dict(),
            rtol=0,
            atol=0,
        )
        meta = DraftVocabulary(vocab_size=8, draft_vocab_size=4, device="meta")
        self.assertTrue(all(t.is_meta for t in meta.buffers()))
        with self.assertRaisesRegex(ValueError, "materialize"):
            install(meta)
        meta.to_empty(device="cpu")
        meta.reset_draft_parameters()
        self.assertFalse(meta.ready)
        install(meta)
        meta.to(dtype=torch.bfloat16)
        self.assertEqual(meta.d2t.dtype, torch.int64)
        self.assertEqual(meta.t2d.dtype, torch.bool)

    def test_restore_rejects_missing_or_inconsistent_mapping(self):
        owner = DFlashDraftModel(model_config())
        install(owner.vocabulary)
        self.assertNotIn("ready", owner.vocabulary.state_dict())
        for missing in (True, False):
            with self.subTest(missing=missing), TemporaryDirectory() as path:
                owner.save_pretrained(path)
                filename = str(Path(path) / "model.safetensors")
                state = load_file(filename)
                if missing:
                    del state["vocabulary.d2t"]
                else:
                    state["vocabulary.d2t"].zero_()
                save_file(state, filename, metadata={"format": "pt"})
                with self.assertRaises(ValueError):
                    DFlashDraftModel.from_pretrained(path)
                with self.assertRaises(ValueError):
                    owner.load_state_dict(state)
                self.assertFalse(owner.vocabulary.ready)
                install(owner.vocabulary)

    def test_partial_load_preserves_mapping_and_full_load_installs_it(self):
        owner = DFlashDraftModel(model_config())
        install(owner.vocabulary)
        expected = owner.vocabulary.to_target(torch.arange(4))
        owner.load_state_dict({"projection.weight": torch.eye(4)}, strict=False)
        torch.testing.assert_close(
            owner.vocabulary.to_target(torch.arange(4)), expected
        )
        restored = DraftVocabulary(vocab_size=8, draft_vocab_size=4)
        restored.load_state_dict(owner.vocabulary.state_dict())
        torch.testing.assert_close(restored.to_target(torch.arange(4)), expected)

    def test_replaced_storage_invalidates_mapping(self):
        vocabulary = DraftVocabulary(vocab_size=8, draft_vocab_size=4)
        install(vocabulary)
        vocabulary.to("meta").to_empty(device="cpu")
        self.assertFalse(vocabulary.ready)
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            vocabulary.to_target(torch.arange(4))
        install(vocabulary)
        self.assertTrue(vocabulary.ready)

    def test_unmapped_checkpoint_restores_without_enabling_logits(self):
        owner = DFlashDraftModel(model_config())
        with TemporaryDirectory() as path:
            owner.save_pretrained(path)
            restored = DFlashDraftModel.from_pretrained(path)
        self.assertFalse(restored.vocabulary.ready)
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            restored.vocabulary.to_target(torch.arange(4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cpu_mapping_can_be_installed_on_cuda(self):
        vocabulary = DraftVocabulary(vocab_size=8, draft_vocab_size=4, device="cuda")
        install(vocabulary)
        result = vocabulary.to_target(torch.arange(4, device="cuda"))
        torch.testing.assert_close(
            result, torch.tensor([0, 2, 5, 7], device="cuda"), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
