# coding=utf-8
"""``model.target_head_path`` routes trainer-side head loading to another checkpoint."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from specforge.algorithms import model_providers
from specforge.training import provenance


def _cfg(target_head_path=None) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            target_model_path="/models/target-nvfp4",
            target_head_path=target_head_path,
            embedding_key="model.embed_tokens.weight",
            lm_head_key="lm_head.weight",
            cache_dir=None,
            trust_remote_code=False,
            mask_token_id=7,
            torch_dtype="bfloat16",
        ),
        training=SimpleNamespace(
            attention_backend="sdpa",
            num_anchors=4,
            loss_decay_gamma=1.0,
            objective_chunk_blocks=0,
        ),
    )


class _Draft:
    block_size = 4
    target_layer_ids = (1, 2)

    def __init__(self):
        self.config = SimpleNamespace(dflash_config={}, vocab_size=16)
        self.mask_token_id = None


class _Built:
    def to(self, **_kwargs):
        return self


class TargetHeadPathTest(unittest.TestCase):
    def _build(self, cfg):
        parts = SimpleNamespace(lm_head="lm_head", embed_tokens="embed")
        with mock.patch(
            "specforge.modeling.target.target_utils.TargetEmbeddingsAndHead.from_pretrained",
            return_value=parts,
        ) as from_pretrained:
            model_providers._build_dflash_family_model(
                cfg, _Draft(), tokenizer=None, model_factory=lambda common: _Built()
            )
        return from_pretrained

    def test_dflash_family_head_defaults_to_the_target_checkpoint(self):
        from_pretrained = self._build(_cfg())
        self.assertEqual(from_pretrained.call_args.args, ("/models/target-nvfp4",))

    def test_dflash_family_head_reads_the_configured_head_checkpoint(self):
        from_pretrained = self._build(_cfg(target_head_path="/models/target-bf16-head"))
        self.assertEqual(from_pretrained.call_args.args, ("/models/target-bf16-head",))
        self.assertEqual(
            from_pretrained.call_args.kwargs["lm_head_key"], "lm_head.weight"
        )


class TargetHeadProvenanceTest(unittest.TestCase):
    def _provenance(self, cfg):
        return provenance._compute_model_resume_provenance(
            cfg,
            SimpleNamespace(_commit_hash=None),
            SimpleNamespace(_commit_hash=None),
            capture_layers=[1, 2],
        )

    def _cfg(self, target_head_path):
        cfg = _cfg(target_head_path)
        cfg.model.draft_model_config = "remote/draft"
        cfg.model.vocab_mapping_path = ""
        cfg.model.load_target_embedding = True
        cfg.model.input_modality = "text"
        return cfg

    def test_unset_head_path_keeps_the_existing_contract_shape(self):
        mapping = self._provenance(self._cfg(None))
        self.assertNotIn("target_head", mapping)

    def test_head_path_is_recorded_with_its_source_identity(self):
        with TemporaryDirectory(prefix="target-head-") as directory:
            head = Path(directory)
            (head / "config.json").write_text("{}", encoding="utf-8")
            (head / "model.safetensors").write_bytes(b"head")
            mapping = self._provenance(self._cfg(str(head)))
            self.assertEqual(
                mapping["target_head"], provenance.model_source_identity(str(head))
            )
            self.assertNotEqual(mapping["target_head"], mapping["target_model"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
