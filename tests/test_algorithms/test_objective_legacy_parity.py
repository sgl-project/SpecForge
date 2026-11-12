"""Compare canonical training losses and gradients with the legacy wrappers."""

import copy
import importlib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDominoModel,
    OnlineDSparkModel,
)
from specforge.algorithms.dspine.model import OnlineDSpineModel
from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import (
    dflash2_config,
    domino_config,
    draft_config,
    dspark_config,
    dspine_config,
    training_batch,
)
from tests.test_modeling.legacy_fixtures import legacy_dflash_config
from tests.test_modeling.legacy_rope_reference import legacy_fp32_rope
from tests.test_modeling.test_canonical_dflash import legacy_name


def block_pair(kind, *, shifted=False):
    configs = {
        "dflash": lambda: draft_config(target_bindings=(), target_logit_transform=None),
        "dflash2": dflash2_config,
        "domino": lambda: domino_config(shifted=shifted),
        "dspark": dspark_config,
        "dspine": dspine_config,
    }
    config = configs[kind]()
    config = replace(
        config,
        spec_config=replace(
            config.spec_config, block=replace(config.spec_config.block, block_size=4)
        ),
    )
    model = build_draft_model(config)
    for name in config.target_bindings:
        model.get_submodule(name).load_target_weight(
            torch.randn(config.vocab_size, config.hidden_size)
        )
    if kind == "dspine":
        model.initialize_transfer_space()
    options = {"target_layer_ids": [1, 3], "mask_token_id": 63}
    options.update(
        {
            "dflash": {},
            "dflash2": dict(
                conv_kernel_size=3,
                conv_group_size=4,
                selector_rank=4,
                selector_top_k=3,
                output_multiplier=1.7,
                final_logit_softcapping=2.5,
            ),
            "domino": dict(
                emb_dim=8,
                gru_hidden_dim=16,
                pure_draft_prefix_len=1,
                shift_label=shifted,
            ),
            "dspark": dict(
                markov_rank=4,
                markov_head_type="rnn",
                enable_confidence_head=True,
                confidence_head_with_markov=True,
            ),
            "dspine": {},
        }[kind]
    )
    legacy_config = legacy_dflash_config(
        block_size=4,
        dflash_config=options,
        **(
            {"dspine_config": dict(transfer_rank=8, message_size=4)}
            if kind == "dspine"
            else {}
        ),
    )
    cls = getattr(
        importlib.import_module(f"specforge.modeling.draft.{kind}"), config.architecture
    )
    reference = cls(legacy_config)

    def name_for(name):
        return legacy_name(name) or (
            name.removeprefix("gru_head.") if kind == "domino" else name
        )

    reference.load_state_dict(
        {
            name_for(n): v
            for n, v in model.state_dict().items()
            if n not in ("embed_tokens.weight", "lm_head.weight")
        }
    )
    wrapper = {
        "dflash": OnlineDFlashModel,
        "dflash2": OnlineDFlashModel,
        "domino": OnlineDominoModel,
        "dspark": OnlineDSparkModel,
        "dspine": OnlineDSpineModel,
    }[kind]
    old = wrapper(
        draft_model=reference,
        target_lm_head=copy.deepcopy(model.lm_head),
        target_embed_tokens=copy.deepcopy(model.embed_tokens),
        mask_token_id=63,
        block_size=4,
        num_anchors=2,
        attention_backend="sdpa",
        fused_logits_loss=False,
        objective_checkpointing=False,
        **({"shift_label": shifted} if kind == "domino" else {}),
    )
    new = build_objective_model(
        model, options={"num_anchors": 2}, target_head=copy.deepcopy(model.lm_head)
    )
    parameter_names = {
        n: (
            n
            if n in ("embed_tokens.weight", "lm_head.weight")
            else "draft_model." + name_for(n)
        )
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    return new, old, parameter_names


def assert_loss_and_gradients(new, old, new_loss, old_loss, names):
    torch.testing.assert_close(new_loss, old_loss, rtol=2e-5, atol=2e-6)
    new_loss.backward()
    old_loss.backward()
    parameters = dict(old.named_parameters())
    for name, parameter in new.draft_model.named_parameters():
        if name in names:
            torch.testing.assert_close(
                parameter.grad,
                parameters[names[name]].grad,
                rtol=2e-4,
                atol=2e-6,
                msg=lambda message: f"{name}: {message}",
            )


class LegacyObjectiveParityTest(unittest.TestCase):
    @legacy_fp32_rope()
    def test_block_losses_and_gradients(self):
        for kind, shifted in (
            ("dflash", False),
            ("dflash2", False),
            ("dspark", False),
            ("domino", False),
            ("domino", True),
            ("dspine", False),
        ):
            for step, evaluation in ((0, False), (5, False), (1, True)):
                with self.subTest(
                    kind=kind, shifted=shifted, step=step, evaluation=evaluation
                ):
                    torch.manual_seed(7)
                    new, old, names = block_pair(kind, shifted=shifted)
                    new.draft_model.train(not evaluation)
                    old.train(not evaluation)
                    batch = training_batch(new.draft_model.config.draft)
                    batch["loss_mask"][0, 4:6] = 0
                    old_args = {
                        key: batch[key]
                        for key in ("input_ids", "hidden_states", "loss_mask")
                    }
                    if kind in ("dspark", "dspine"):
                        old_args["target_last_hidden_states"] = batch[
                            "target_last_hidden_states"
                        ].detach()
                    if kind == "domino":
                        old_args["lambda_base"] = max(0, 1 - step / 5)  # int(11 * .5)
                    if kind == "dspine":
                        old_args.update(global_step=step, total_steps=11)
                    if kind == "dflash2":
                        new.objective.selector_warmup_steps = 1
                        new.objective.selector_ramp_steps = 5
                        old_args["selector_loss_alpha"] = min(max(step / 5, 0), 1)
                    with patch.dict(
                        "os.environ", {"SPECFORGE_DFLASH2_FUSED_CONV": "0"}
                    ):
                        torch.manual_seed(11)
                        result = new(batch, step=step, total_steps=11)
                        rng = torch.get_rng_state()
                        torch.manual_seed(11)
                        expected = old(**old_args)[0]
                    torch.testing.assert_close(torch.get_rng_state(), rng)
                    assert_loss_and_gradients(new, old, result.loss, expected, names)

    @legacy_fp32_rope()
    def test_eagle_losses_gradients_and_microbatch_means(self):
        from contextlib import ExitStack
        from types import SimpleNamespace

        from transformers import LlamaConfig

        from specforge.modeling.draft.peagle import PEagleDraftModel
        from tests.test_algorithms.test_objectives import eagle_config, peagle_config
        from tests.test_modeling.test_canonical_eagle3 import (
            install_mapping,
            legacy_model,
        )
        from tests.test_modeling.test_canonical_eagle3 import legacy_name as eagle_name
        from tests.test_modeling.test_canonical_peagle import legacy_name as peagle_name
        from tests.test_modeling.test_canonical_peagle import peagle_eager_reference

        # Only replace the GPU-only loss kernel; keep the legacy rollout and reductions.
        def soft_ce(logits, target, mask, acceptance_target=None):
            loss = (
                -(logits.float().log_softmax(-1) * target).sum(-1) * mask.squeeze(-1)
            ).mean()
            if acceptance_target is None:
                return loss
            return loss, torch.minimum(
                logits.float().softmax(-1), acceptance_target
            ).sum(-1)

        kernel = SimpleNamespace(LogSoftmaxLoss=SimpleNamespace(apply=soft_ce))

        def legacy_module(kind):
            path = (
                Path(__file__).resolve().parents[2]
                / "specforge"
                / "algorithms"
                / kind
                / "model.py"
            )
            spec = importlib.util.spec_from_file_location(f"reference_{kind}", path)
            module = importlib.util.module_from_spec(spec)
            with patch.dict("sys.modules", {"specforge.core.loss": kernel}):
                spec.loader.exec_module(module)
            return module

        eagle, peagle = legacy_module("eagle3"), legacy_module("peagle")

        def dense_mask(mask_mod, *, Q_LEN, KV_LEN, **kwargs):
            return mask_mod(
                0, 0, torch.arange(Q_LEN)[:, None], torch.arange(KV_LEN)[None]
            )[None, None]

        with ExitStack() as stack:
            stack.enter_context(patch.object(peagle, "create_block_mask", dense_mask))
            stack.enter_context(
                patch(
                    "specforge.modeling.draft.peagle.compile_friendly_flex_attention",
                    peagle_eager_reference,
                )
            )
            for kind in ("eagle3", "peagle"):
                results, expected_losses = [], []
                for length in (7, 9):
                    with self.subTest(kind=kind, length=length):
                        torch.manual_seed(7)
                        config = eagle_config() if kind == "eagle3" else peagle_config()
                        model = build_draft_model(config)
                        install_mapping(model)
                        if kind == "eagle3":
                            reference = legacy_model(model)
                            old = eagle.OnlineEagle3Model(reference, length=3)
                            names = {
                                n: "draft_model." + eagle_name(n)
                                for n, _ in model.named_parameters()
                            }
                        else:
                            old_config = LlamaConfig(
                                hidden_size=32,
                                target_hidden_size=48,
                                intermediate_size=64,
                                num_attention_heads=4,
                                num_key_value_heads=2,
                                head_dim=8,
                                num_hidden_layers=3,
                                vocab_size=64,
                                draft_vocab_size=32,
                                max_position_embeddings=128,
                                rms_norm_eps=1e-6,
                                hidden_act="silu",
                                pretraining_tp=1,
                                norm_before_residual=False,
                                rope_parameters={
                                    "rope_type": "default",
                                    "rope_theta": 10000.0,
                                },
                            )
                            reference = PEagleDraftModel(old_config)
                            reference.load_state_dict(
                                {
                                    peagle_name(n): v
                                    for n, v in model.state_dict().items()
                                }
                            )
                            old = peagle.OnlinePEagleModel(
                                reference,
                                mask_token_id=config.spec_config.mask_token_id,
                                num_depths=3,
                                down_sample_ratio=0.8,
                            )
                            names = {
                                n: "draft_model." + peagle_name(n)
                                for n, _ in model.named_parameters()
                            }
                        options = (
                            {"ttt_length": 3} if kind == "eagle3" else {"num_depths": 3}
                        )
                        new = build_objective_model(model, options=options)
                        batch = {
                            k: v[:, :length].detach()
                            for k, v in training_batch(config).items()
                        }
                        batch["target_logits"] = torch.randn(
                            1, length, config.vocab_size
                        )
                        batch["loss_mask"][:, -1] = 0
                        shift = lambda value: torch.cat(
                            (value[:, 1:], torch.zeros_like(value[:, :1])), 1
                        )
                        old_args = dict(
                            input_ids=shift(batch["input_ids"]),
                            attention_mask=batch["attention_mask"],
                            target=shift(batch["target_logits"]),
                            loss_mask=shift(batch["loss_mask"]),
                            hidden_states=batch["hidden_states"],
                        )
                        if kind == "eagle3":
                            old_args["loss_mask"] = old_args["loss_mask"][..., None]
                        torch.manual_seed(11)
                        result = new(batch, step=0, total_steps=None)
                        torch.manual_seed(11)
                        expected = old(**old_args)[0]
                        if kind == "eagle3":
                            expected = sum(
                                0.8**depth * loss for depth, loss in enumerate(expected)
                            )
                        assert_loss_and_gradients(
                            new, old, result.loss, expected, names
                        )
                        results.append(result)
                        expected_losses.append(expected.detach())
                pooled = sum(r.numerator.detach() for r in results) / sum(
                    r.denominator for r in results
                )
                torch.testing.assert_close(pooled, torch.stack(expected_losses).mean())

    def test_mtp_matches_legacy_wrapper_with_native_layers(self):
        from types import SimpleNamespace

        from torch import nn

        from specforge.config.draft.norm import RMSNormSpec
        from specforge.config.draft.rope import DefaultRoPESpec
        from specforge.core.mtp import OnlineMTPModel
        from tests.test_config.test_draft_mtp_spec import mtp_config
        from tests.test_modeling.test_canonical_mtp import (
            reference_forward,
            reference_model,
        )

        config = mtp_config(target_bindings=(), target_logit_transform=None)
        config = replace(
            config,
            norm_config=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=1),
            attention_configs={
                "full": replace(config.attention_configs["full"], output_gate=True)
            },
            rope_parameters=DefaultRoPESpec(
                interleaved=False, rope_theta=10000000, partial_rotary_factor=0.5
            ),
        )
        model = build_draft_model(config)
        adapter = nn.Module()
        adapter.mtp = reference_model(config)
        adapter.mtp.load_state_dict(model.state_dict())
        adapter.config = SimpleNamespace(pad_token_id=0)
        adapter.forward_hidden = lambda input_ids, hidden_states, attention_mask, position_ids: reference_forward(
            adapter.mtp, input_ids, hidden_states, position_ids
        )[
            0
        ]
        old = OnlineMTPModel(adapter, objective_chunk_size=0)
        new = build_objective_model(model)
        batch = training_batch(config)
        batch["attention_mask"].fill_(1)
        actual = new(batch, step=0, total_steps=None).loss
        expected = old(
            input_ids=batch["input_ids"],
            hidden_states=batch["target_last_hidden_states"].detach(),
            loss_mask=batch["loss_mask"],
            attention_mask=batch["attention_mask"],
        )[0]
        assert_loss_and_gradients(
            new,
            old,
            actual,
            expected,
            {n: "draft_model.mtp." + n for n, _ in model.named_parameters()},
        )

    def test_dmtp_matches_explicit_suffix_loss(self):
        # DMTP has no legacy Online wrapper. Check its label alignment independently.
        from specforge.algorithms.objectives.inputs import gather
        from tests.test_algorithms.test_objectives import dmtp_config

        config = dmtp_config()
        model = build_draft_model(config)
        reference = copy.deepcopy(model)
        batch = training_batch(config)
        batch["loss_mask"][0, 4] = 0
        calls = []
        model.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append((args, kwargs)), with_kwargs=True
        )
        runtime = build_objective_model(model, options={"num_anchors": 2})
        actual = runtime(batch, step=0, total_steps=None).loss
        args, kwargs = calls[0]
        logits = reference(*args, **{**kwargs, "output_logits": True}).logits.reshape(
            1, -1, 4, config.vocab_size
        )
        # Fixture tokens equal positions; the first input is the external candidate.
        first_positions = args[0].reshape(1, -1, 4)[..., 0]
        positions = first_positions[..., None] + torch.arange(4)
        labels = gather(batch["input_ids"], positions)
        mask = gather(batch["loss_mask"], positions).bool() & (positions < 9)
        mask[..., 0] = False
        ce = torch.nn.functional.cross_entropy(
            logits.flatten(0, 2), labels.flatten(), reduction="none"
        ).reshape_as(labels)
        expected = (ce * mask).sum() / mask.sum()
        assert_loss_and_gradients(
            runtime,
            reference,
            actual,
            expected,
            {n: n for n, _ in model.named_parameters()},
        )
