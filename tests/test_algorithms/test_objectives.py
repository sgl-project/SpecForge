"""Default objective coverage and independently checked token alignment."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from specforge.algorithms.objectives import (
    build_objective_model,
    objective_registrations,
)
from specforge.algorithms.objectives.inputs import prepare_blocks
from specforge.modeling import build_draft_model
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_modeling.test_canonical_dflash2 import dflash2_config
from tests.test_modeling.test_canonical_dmtp import model_config as dmtp_config
from tests.test_modeling.test_canonical_domino import model_config as domino_config
from tests.test_modeling.test_canonical_dspark import model_config as dspark_config
from tests.test_modeling.test_canonical_dspine import model_config as dspine_config
from tests.test_modeling.test_canonical_eagle3 import model_config as eagle_config
from tests.test_modeling.test_canonical_peagle import model_config as peagle_config


def configurations():
    return [
        draft_config(target_bindings=(), target_logit_transform=None),
        dflash2_config(),
        dspark_config(),
        domino_config(),
        dspine_config(),
        dmtp_config(),
        mtp_config(target_bindings=(), target_logit_transform=None),
        eagle_config(),
        peagle_config(),
    ]


def training_batch(config, device="cpu"):
    spec = config.spec_config
    width = (
        3 * spec.target_features.hidden_size
        if config.architecture in ("Eagle3DraftModel", "PEagleDraftModel")
        else 64
    )
    final_width = (
        spec.target_features.hidden_size
        if config.architecture == "MTPDraftModel"
        else config.hidden_size
    )
    return dict(
        input_ids=torch.arange(9, device=device)[None],
        loss_mask=torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 0]], device=device),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]], device=device),
        hidden_states=torch.randn(1, 9, width, device=device, requires_grad=True),
        target_last_hidden_states=torch.randn(
            1, 9, final_width, device=device, requires_grad=True
        ),
        mtp_hidden_states=torch.randn(1, 9, 48, device=device, requires_grad=True),
    )


class ObjectiveTest(unittest.TestCase):
    def test_defaults_train_every_serving_parameter(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for config in configurations():
                with self.subTest(device=device, architecture=config.architecture):
                    torch.manual_seed(11)
                    model = build_draft_model(config, device=device)
                    if hasattr(model, "vocabulary"):
                        selected = torch.arange(config.draft_vocab_size, device=device)
                        model.vocabulary.load_mapping(
                            t2d=torch.arange(config.vocab_size, device=device)
                            < config.draft_vocab_size,
                            d2t=torch.zeros_like(selected),
                        )
                    for name in config.target_bindings:
                        model.get_submodule(name).load_target_weight(
                            torch.randn(
                                config.vocab_size, config.hidden_size, device=device
                            )
                        )
                    if hasattr(model, "initialize_transfer_space"):
                        model.initialize_transfer_space()
                    options = {"num_anchors": 2} if config.spec_config.protocol else {}
                    if config.architecture == "Eagle3DraftModel":
                        options["ttt_length"] = 2
                    if config.architecture == "PEagleDraftModel":
                        options.update(num_depths=2)
                    runtime = build_objective_model(
                        model,
                        options=options,
                        target_head=nn.Linear(
                            32, config.vocab_size, bias=False, device=device
                        ),
                    )
                    batch = training_batch(config, device)
                    result = runtime(batch, step=1, total_steps=10)
                    self.assertTrue(torch.isfinite(result.loss))
                    result.loss.backward()
                    self.assertEqual(
                        [
                            n
                            for n, p in model.named_parameters()
                            if p.requires_grad
                            and (p.grad is None or not torch.isfinite(p.grad).all())
                        ],
                        [],
                    )
                    self.assertTrue(
                        all(p.grad is None for p in runtime.target_head.parameters())
                    )
                    self.assertTrue(
                        all(t.grad is None for t in batch.values() if t.requires_grad)
                    )
        self.assertEqual(len(objective_registrations()), 9)

    def test_block_alignment_and_empty_supervision(self):
        config = dmtp_config()
        batch = training_batch(config)
        blocks = prepare_blocks(
            config,
            batch,
            num_anchors=2,
            input_offset=1,
            label_offset=1,
            context_offset=-1,
        )
        self.assertTrue(
            torch.equal(
                blocks.input_ids.reshape_as(blocks.labels)[..., 0], blocks.anchors + 1
            )
        )
        self.assertTrue(
            torch.equal(
                blocks.label_positions, blocks.anchors[..., None] + 1 + torch.arange(4)
            )
        )
        self.assertFalse(blocks.weights[..., 0].any())
        from specforge.modeling.draft.layers.attention.masks import dense_attention_mask

        visible = dense_attention_mask(
            blocks.layer_inputs[0].attention_layout,
            is_causal=False,
            sliding_window=None,
        )[0, 0]
        for index, anchor in enumerate(blocks.anchors[0]):
            self.assertFalse(visible[index * 4, anchor:9].any())
        self.assertFalse(visible[:4, 13:].any())
        batch["loss_mask"].zero_()
        model = build_draft_model(config)
        result = build_objective_model(model, options={"num_anchors": 2})(batch)
        self.assertEqual(result.loss.item(), 0)
        result.loss.backward()

    def test_mtp_matches_shifted_cross_entropy(self):
        config = mtp_config(target_bindings=(), target_logit_transform=None)
        model = build_draft_model(config)
        batch = training_batch(config)
        captured = []
        handle = model.lm_head.register_forward_hook(
            lambda module, args, output: captured.append(output)
        )
        result = build_objective_model(model)(batch)
        handle.remove()
        labels = batch["input_ids"][:, 2:8]
        reference = torch.nn.functional.cross_entropy(captured[0][:6], labels.flatten())
        torch.testing.assert_close(result.loss, reference)

    def test_domino_reuses_base_logits_and_aligned_predecessors(self):
        for shifted in (False, True):
            config = domino_config(shifted=shifted)
            model = build_draft_model(config)
            batch = training_batch(config)
            blocks = prepare_blocks(
                config, batch, num_anchors=2, label_offset=int(shifted)
            )
            runtime = build_objective_model(
                model, options={"num_anchors": 2, "objective_checkpointing": False}
            )
            with (
                patch(
                    "specforge.algorithms.objectives.domino.prepare_blocks",
                    return_value=blocks,
                ),
                patch.object(
                    model.embed_tokens, "forward", wraps=model.embed_tokens.forward
                ) as embedding,
                patch.object(
                    model.lm_head, "forward", wraps=model.lm_head.forward
                ) as projection,
            ):
                result = runtime(batch, step=1, total_steps=10)
                result.loss.backward()
                self.assertEqual(projection.call_count, 1)
                torch.testing.assert_close(
                    embedding.call_args.args[0], blocks.predecessors
                )
            with torch.no_grad():
                output = model(
                    blocks.input_ids,
                    batch["hidden_states"],
                    blocks.layer_inputs,
                    predecessor_ids=blocks.predecessors.flatten(1),
                )
            labels, weights = blocks.labels.flatten(), blocks.weights.flatten()

            def ce(logits):
                return torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), labels, reduction="none"
                )

            expected = (
                (0.8 * ce(output.base_logits) + 0.2 * ce(output.logits)) * weights
            ).sum() / (weights.sum() + 1e-6)
            torch.testing.assert_close(result.loss, expected)
            self.assertIsNone(output.correction_logits)

    def test_reject_incompatible_or_unknown_options(self):
        model = build_draft_model(dflash2_config())
        with self.assertRaisesRegex(ValueError, "does not support"):
            build_objective_model(model, name="dflash")
        with self.assertRaises(TypeError):
            build_objective_model(model, options={"unknown": True})
