import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDominoModel,
    OnlineDSparkModel,
)
from specforge.algorithms.common.target_kv_data import CAPTURE_FORMAT
from specforge.application.composition import resolve_run
from specforge.config import Config
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.target_kv import (
    capture_geometry,
    resolve_target_kv_config,
    target_kv_config,
)
from specforge.runtime.contracts import TrainBatch
from specforge.training.model_loading import resolve_draft_config

VARIANTS = (
    ("dflash", DFlashDraftModel, OnlineDFlashModel),
    ("dflash", DFlash2DraftModel, OnlineDFlashModel),
    ("domino", DominoDraftModel, OnlineDominoModel),
    ("dspark", DSparkDraftModel, OnlineDSparkModel),
)


def target_config():
    return Qwen3Config(
        hidden_size=16,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=3,
        layer_types=["full_attention"] * 3,
    )


def draft_config(algorithm, architecture, mode="derope_reproject"):
    options = dict(
        attention_mode="gqa",
        projector_type=algorithm,
        mask_token_id=31,
        target_layer_ids=[0, 2],
        target_kv_position_mode=mode,
        target_kv_context_key_norm=mode == "derope_reproject",
    )
    if algorithm == "dspark":
        options.update(markov_rank=4, enable_confidence_head=True)
    if algorithm == "domino":
        options.update(emb_dim=8, gru_hidden_dim=8)
    if architecture == "DFlash2DraftModel":
        options.update(
            conv_kernel_size=2, conv_group_size=4, selector_rank=4, selector_top_k=3
        )
    cfg = Qwen3Config(
        architectures=[architecture],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=2,
        num_target_layers=3,
        vocab_size=32,
        layer_types=["full_attention"] * 2,
        block_size=4,
        is_causal=True,
        dflash_config=options,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def run_config(algorithm, **model):
    return Config.model_validate(
        dict(
            model=dict(
                target_model_path="unused-target",
                draft_model_config="unused-draft",
                **model,
            ),
            data=dict(hidden_states_path="unused-features"),
            training=dict(strategy=algorithm, attention_backend="sdpa"),
        )
    )


def wrapper(cls, draft):
    return cls(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(16, 32, bias=False).requires_grad_(False),
        target_embed_tokens=torch.nn.Embedding(32, 16).requires_grad_(False),
        mask_token_id=31,
        block_size=4,
        num_anchors=2,
        attention_backend="sdpa",
        objective_chunk_blocks=1,
    )


class KVConditioningTest(unittest.TestCase):
    def test_legacy_recipe_normalizes_to_the_single_dspark_algorithm(self):
        registry = builtin_algorithm_registry()
        self.assertNotIn("dspark_kv", registry.names)
        raw = run_config("dspark").model_dump()
        raw["training"]["strategy"] = "dspark_kv"
        del raw["model"]["conditioning_source"]
        original = copy.deepcopy(raw)
        migrated = Config.model_validate(raw)
        self.assertEqual(raw, original)
        canonical = run_config("dspark", conditioning_source="kv")
        self.assertEqual(migrated.model_dump(), canonical.model_dump())
        resolved = resolve_run(migrated).algorithm
        self.assertEqual(resolved.name, "dspark")
        self.assertEqual(resolved.providers.algorithm_name, "dspark")
        from specforge.training.strategies.base import DSparkTrainStrategy

        self.assertIs(type(resolved.providers.step.build(None)), DSparkTrainStrategy)
        raw["model"]["conditioning_source"] = "hidden"
        with self.assertRaisesRegex(ValueError, "legacy dspark_kv"):
            Config.model_validate(raw)

    def test_hidden_default_and_explicit_hidden_keep_catalog_and_outputs(self):
        registry = builtin_algorithm_registry()
        for algorithm, draft_cls, model_cls in VARIANTS:
            with self.subTest(model=draft_cls.__name__):
                cfg = run_config(algorithm)
                self.assertEqual(cfg.model.conditioning_source, "hidden")
                self.assertIs(
                    resolve_run(cfg, registry).algorithm, registry.resolve(algorithm)
                )
                cfg.model.conditioning_source = "hidden"
                self.assertIs(
                    resolve_run(cfg, registry).algorithm, registry.resolve(algorithm)
                )
                raw = draft_config(algorithm, draft_cls.__name__)
                raw.dflash_config = {
                    k: v
                    for k, v in raw.dflash_config.items()
                    if not k.startswith("target_kv_")
                }
                models = []
                for explicit in (False, True):
                    options = copy.deepcopy(raw)
                    if explicit:
                        options.conditioning_source = "hidden"
                    torch.manual_seed(19)
                    models.append(wrapper(model_cls, draft_cls(options)))
                self.assertEqual(
                    models[0].state_dict().keys(), models[1].state_dict().keys()
                )
                tokens = torch.randint(0, 30, (2, 12))
                args = dict(
                    input_ids=tokens,
                    hidden_states=torch.randn(2, 12, 32),
                    loss_mask=torch.ones_like(tokens),
                )
                if algorithm != "domino":
                    args["target_last_hidden_states"] = torch.randn(2, 12, 16)
                outputs = []
                for model in models:
                    torch.manual_seed(12)
                    outputs.append(model(**args))
                torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)

    def test_all_variants_kv_backward_frozen_teacher_and_reload(self):
        for algorithm, draft_cls, model_cls in VARIANTS:
            for mode in ("cached", "derope_reproject"):
                with self.subTest(model=draft_cls.__name__, mode=mode):
                    torch.manual_seed(42)
                    raw = draft_config(algorithm, draft_cls.__name__, mode)
                    resolve_target_kv_config(raw, target_config())
                    draft = draft_cls(raw)
                    model = wrapper(model_cls, draft)
                    resolved = resolve_run(
                        run_config(algorithm, conditioning_source="kv")
                    )
                    step = resolved.algorithm.providers.step.build(model)
                    tensors = dict(
                        input_ids=torch.randint(0, 30, (2, 12)),
                        loss_mask=torch.ones(2, 12),
                        target_kv=torch.randn(2, 12, 2, 2, 2, 4, requires_grad=True),
                    )
                    if algorithm != "domino":
                        tensors["target_last_hidden_states"] = torch.randn(
                            2, 12, 16, requires_grad=True
                        )
                    result = step.forward_loss(
                        TrainBatch(
                            sample_ids=["a", "b"], strategy=algorithm, tensors=tensors
                        )
                    )
                    self.assertTrue(torch.isfinite(result.loss))
                    result.loss.backward()
                    self.assertFalse(hasattr(draft, "fc"))
                    for layer in draft.layers:
                        for projection in (
                            layer.self_attn.target_k_proj,
                            layer.self_attn.target_v_proj,
                        ):
                            self.assertIsNotNone(projection.weight.grad)
                            self.assertTrue(
                                torch.isfinite(projection.weight.grad).all()
                            )
                            self.assertGreater(
                                projection.weight.grad.abs().sum().item(), 0
                            )
                    for tensor in tensors.values():
                        self.assertIsNone(tensor.grad)
                    head_name = {
                        "DFlash2DraftModel": "candidate_selector",
                        "DominoDraftModel": "prefix_gru",
                        "DSparkDraftModel": "markov_head",
                    }.get(draft_cls.__name__)
                    if head_name:
                        gradients = [
                            p.grad for p in getattr(draft, head_name).parameters()
                        ]
                        self.assertTrue(
                            all(
                                g is not None and torch.isfinite(g).all()
                                for g in gradients
                            )
                        )
                        self.assertGreater(
                            sum(g.abs().sum().item() for g in gradients), 0
                        )
                    for name in ("lm_head", "embed_tokens"):
                        self.assertTrue(
                            all(
                                p.grad is None
                                for p in getattr(model, name).parameters()
                            )
                        )
                    contract = resolved.algorithm.providers.step.resume_contract(
                        resolved.config, draft, model
                    )
                    self.assertEqual(contract["conditioning_source"], "kv")
                    self.assertEqual(
                        contract["target_kv_geometry"], target_kv_config(raw)
                    )
                    with tempfile.TemporaryDirectory() as path:
                        draft.save_pretrained(path)
                        # No target repo/config load is permitted on this path.
                        with patch(
                            "specforge.modeling.target.target_utils.load_target_config",
                            side_effect=AssertionError("target access"),
                        ):
                            loaded = AutoDraftModel.from_pretrained(
                                path, attn_implementation="sdpa"
                            )
                        self.assertEqual(loaded.config.conditioning_source, "kv")
                        self.assertEqual(type(loaded), type(draft))
                        for name, weight in draft.state_dict().items():
                            torch.testing.assert_close(
                                loaded.state_dict()[name], weight, rtol=0, atol=0
                            )

    def test_resolution_derives_geometry_and_rejects_stale_saved_values(self):
        for algorithm, draft_cls, _ in VARIANTS:
            with (
                self.subTest(model=draft_cls.__name__),
                tempfile.TemporaryDirectory() as path,
            ):
                raw = draft_config(algorithm, draft_cls.__name__)
                config_path = Path(path) / "draft.json"
                raw.to_json_file(config_path)
                cfg = run_config(algorithm, conditioning_source="kv")
                cfg.model.draft_model_config = str(config_path)
                provider = resolve_run(cfg).algorithm.providers.model.draft_config
                with patch(
                    "specforge.modeling.target.target_utils.load_target_config",
                    return_value=target_config(),
                ):
                    resolved = resolve_draft_config(cfg, provider=provider)
                self.assertEqual(resolved.conditioning_source, "kv")
                self.assertEqual(target_kv_config(resolved)["heads"], 2)
                self.assertEqual(target_kv_config(resolved)["head_dim"], 4)
                self.assertEqual(resolved.dflash_config["target_layer_ids"], [0, 2])
                for field, value in (
                    ("target_kv_heads", 1),
                    ("target_kv_head_dim", 8),
                    ("target_kv_rope_theta", 3.0),
                    ("target_kv_rotary_dim", 2),
                    ("target_kv_rope_layout", "other"),
                ):
                    bad = copy.deepcopy(resolved)
                    bad.dflash_config[field] = value
                    with self.assertRaisesRegex(ValueError, "disagrees"):
                        resolve_target_kv_config(bad, target_config())
                resolved.to_json_file(config_path)
                cfg.model.conditioning_source = "hidden"
                with self.assertRaisesRegex(
                    ValueError, "requires model.conditioning_source=kv"
                ):
                    resolve_draft_config(cfg, provider=provider)

    def test_contract_reader_and_collator_preserve_algorithm_identity(self):
        for algorithm in ("dflash", "domino", "dspark"):
            with (
                self.subTest(algorithm=algorithm),
                tempfile.TemporaryDirectory() as directory,
            ):
                run = resolve_run(run_config(algorithm, conditioning_source="kv"))
                offline = run.algorithm.providers.offline_for("text")
                base = builtin_algorithm_registry().resolve(algorithm)
                self.assertEqual(
                    offline.normalizer_id,
                    base.providers.offline_for("text").normalizer_id,
                )
                storage = run.algorithm.spec.feature_contract("offline", "text").storage
                self.assertEqual(storage.normalizer, offline.normalizer_id)
                self.assertEqual(storage.format, CAPTURE_FORMAT)
                normalizer = offline.build_normalizer(max_len=12)
                records = []
                for size in (8, 12):
                    raw = dict(
                        input_ids=torch.arange(size),
                        loss_mask=torch.ones(size),
                        target_kv=torch.randn(size, 2, 2, 2, 4),
                    )
                    if algorithm != "domino":
                        raw["target_last_hidden_states"] = torch.randn(size, 16)
                    torch.save(raw, Path(directory) / f"{size}.ckpt")
                    records.append(normalizer(raw))
                batch = offline.build_collator()(records)
                self.assertEqual(batch["target_kv"].shape, (2, 12, 2, 2, 2, 4))
                self.assertEqual(batch["target_kv"][0, 8:].count_nonzero(), 0)
                reader = offline.build_reader(
                    directory, run_id="test", ttt_length=1, max_len=12
                )
                refs = list(reader)
                self.assertTrue(all(ref.strategy == algorithm for ref in refs))
                contract = run.algorithm.spec.feature_contract("offline", "text")
                self.assertEqual(set(batch), set(contract.required_tensors))
                self.assertNotIn("hidden_states", batch)

    def test_target_head_dimension_fallback_and_failed_resolution_is_atomic(self):
        target = SimpleNamespace(
            **{k: v for k, v in target_config().to_dict().items() if k != "head_dim"}
        )
        raw = draft_config("dflash", "DFlashDraftModel")
        resolve_target_kv_config(raw, target)
        self.assertEqual(capture_geometry(raw, target)["head_dim"], 4)
        self.assertEqual(capture_geometry(raw, target)["rotary_dim"], 4)
        invalid = draft_config("dflash", "DFlashDraftModel")
        invalid.dflash_config["target_layer_ids"] = [999]
        original = invalid.to_dict()
        with self.assertRaisesRegex(ValueError, "full-attention"):
            resolve_target_kv_config(invalid, target)
        self.assertEqual(invalid.to_dict(), original)

    def test_invalid_sources_and_unsupported_algorithms(self):
        for source in ("target_kv", "other", None):
            with self.assertRaises(ValueError):
                run_config("dspark", conditioning_source=source)
        for algorithm in ("eagle3", "peagle", "mtp"):
            with self.assertRaisesRegex(
                ValueError, "does not support conditioning_source"
            ):
                resolve_run(run_config(algorithm, conditioning_source="kv"))
        raw = draft_config("dflash", "DFlashDraftModel")
        raw.dflash_config["target_layer_ids"] = [0, 0]
        with self.assertRaises(ValueError):
            resolve_target_kv_config(raw, target_config())
        raw = draft_config("dflash", "DFlashDraftModel")
        target = target_config()
        target.layer_types = ["linear_attention"] * 3
        with self.assertRaisesRegex(ValueError, "full-attention"):
            resolve_target_kv_config(raw, target)

    def test_export_gate_does_not_mislabel_kv_models_as_hidden_consumers(self):
        from scripts.gates.normalize_dflash_export import normalize_export

        for algorithm, draft_cls, _ in VARIANTS:
            with (
                self.subTest(model=draft_cls.__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                raw = draft_config(algorithm, draft_cls.__name__)
                resolve_target_kv_config(raw, target_config())
                # DSpark export normalization requires an explicit Markov type.
                if algorithm == "dspark":
                    raw.dflash_config["markov_head_type"] = "vanilla"
                path = Path(directory) / "config.json"
                raw.to_json_file(path)
                if algorithm == "dspark":
                    result = normalize_export(str(path), 4)
                    self.assertEqual(result["architectures"], ["Qwen3DSparkKVModel"])
                else:
                    original = path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "KV-aware"):
                        normalize_export(str(path), 4)
                    self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
