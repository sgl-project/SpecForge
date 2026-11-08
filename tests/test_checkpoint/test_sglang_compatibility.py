"""Pinned SGLang construction/weight loading and FP32 eager attention oracle.

Run explicitly with SPECFORGE_SGLANG_EXPORT_TESTS=1. Only the paged-attention
backend is replaced by dense attention; serving projections, norms, RoPE,
convolutions and model loops are SGLang's real implementations.
"""

import importlib
import os
import unittest
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import PretrainedConfig

from specforge.checkpoint.formats.sglang import NON_COMPUTATIONAL_READS, convert_config
from specforge.checkpoint.formats.sglang.converters import serving_weights
from specforge.config.draft.norm import RMSNormSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs
from tests.test_checkpoint.test_native_presets import native_configs
from tests.test_checkpoint.test_sglang_export import serving_configs
from tests.test_config.test_draft_mtp_spec import mtp_config


class ReadConfig(PretrainedConfig):
    def __init__(self, fields):
        super().__init__(**fields)
        self._written = set(fields)
        self._reads = set()

    def __getattribute__(self, name):
        try:
            value = super().__getattribute__(name)
        except AttributeError:
            if not name.startswith("_") and "_reads" in self.__dict__:
                self._reads.add(name)
            raise
        if (
            not name.startswith("_")
            and not callable(value)
            and "_reads" in self.__dict__
        ):
            self._reads.add(name)
        return value


def dense_attention(context, positions, nctx, slots):
    from sglang.srt.layers.radix_attention import AttentionType

    def dense(attn, q, k, v, batch, **kwargs):
        if attn.layer_id in context:
            ck, cv = context[attn.layer_id]
            k, v = torch.cat([ck, k]), torch.cat([cv, v])
        q = q.reshape(slots, attn.tp_q_head_num, attn.head_dim).transpose(0, 1)[None]
        k = k.reshape(nctx + slots, attn.tp_k_head_num, attn.head_dim).transpose(0, 1)[
            None
        ]
        v = v.reshape(nctx + slots, attn.tp_v_head_num, attn.head_dim).transpose(0, 1)[
            None
        ]
        distance = positions[:, nctx:, None] - positions[:, None, :]
        visible = torch.ones_like(distance, dtype=torch.bool)
        if attn.attn_type == AttentionType.DECODER:
            visible &= distance >= 0
        if attn.sliding_window_size >= 0:
            visible &= distance.abs() <= attn.sliding_window_size
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=visible[:, None],
            scale=attn.scaling,
            enable_gqa=True,
        )
        return output[0].transpose(0, 1).reshape(slots, -1)

    return dense


@unittest.skipUnless(
    os.environ.get("SPECFORGE_SGLANG_EXPORT_TESTS") == "1",
    "explicit pinned SGLang gate",
)
class SGLangCompatibilityTest(unittest.TestCase):
    def test_real_models_fields_and_outputs(self):
        from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
        from sglang.kernels.spec import KernelBackend
        from sglang.srt.runtime_context import get_context, get_parallel

        old = get_fused_op_backend()
        self.addCleanup(set_fused_op_backend, old)
        set_fused_op_backend(KernelBackend.TORCH)
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"SGLANG_RAGGED_VERIFY_MODE": "cap-accept"})
            )
            stack.enter_context(get_context().override_server_args())
            stack.enter_context(
                get_parallel().override(
                    tp_size=1,
                    tp_rank=0,
                    tp_group=None,
                    pp_rank=0,
                    pp_size=1,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    attn_tp_group=None,
                    pp_group=SimpleNamespace(
                        is_first_rank=True,
                        is_last_rank=True,
                        rank_in_group=0,
                        world_size=1,
                    ),
                )
            )
            stack.enter_context(
                patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
            )
            stack.enter_context(patch("sglang.srt.models.utils._is_cuda", False))
            configs = serving_configs()
            if torch.cuda.is_available():
                configs.extend(native_configs(small=True))
                base = mtp_config()
                configs.append(
                    replace(
                        base,
                        norm_config=RMSNormSpec(
                            has_weight=True, eps=1e-6, weight_offset=1
                        ),
                        attention_configs={
                            "full": replace(
                                base.attention_configs["full"], output_gate=True
                            )
                        },
                    )
                )
            stack.enter_context(patch("sglang.srt.models.qwen3_5._is_cuda", False))
            for config in configs:
                with self.subTest(architecture=config.architecture):
                    self._check_model(config)

    def _check_model(self, config):
        from sglang.srt.layers.radix_attention import RadixAttention

        torch.manual_seed(12)
        fields = convert_config(config)
        native = config.architecture == "MTPDraftModel"
        device = "cuda" if native else "cpu"
        if native:
            from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig

            fields.update(
                model_type="qwen3_5_text",
                full_attention_interval=1,
                num_experts=0,
                architectures=["Qwen3_5ForCausalLMMTP"],
            )
            read = Qwen3_5TextConfig(**fields)
        else:
            read = ReadConfig(fields)
        eagle = config.architecture == "Eagle3DraftModel"
        module = (
            "llama_eagle3"
            if eagle
            else ("dspark" if config.architecture == "DSparkDraftModel" else "dflash")
        )
        if native:
            module = "qwen3_5_mtp"
        cls = getattr(
            importlib.import_module("sglang.srt.models." + module),
            fields["architectures"][0],
        )
        served = cls(read).to(device)
        model = build_draft_model(config, device=device).eval()
        embedding = torch.randn(config.vocab_size, config.hidden_size, device=device)
        model.embed_tokens.load_target_weight(embedding)
        if "lm_head" in config.target_bindings:
            model.lm_head.load_target_weight(
                torch.randn(config.vocab_size, config.hidden_size, device=device)
            )
        if eagle:
            model.vocabulary.load_mapping(
                t2d=torch.arange(config.vocab_size) < config.draft_vocab_size,
                d2t=torch.zeros(config.draft_vocab_size, dtype=torch.long),
            )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if any(
                    part in name
                    for part in (
                        "_conv.",
                        "candidate_selector",
                        "markov_head",
                        "confidence_head",
                        "gru_head",
                    )
                ):
                    parameter.normal_(std=0.1)
        weights = serving_weights(config, model.state_dict())
        if native:
            weights = {"mtp." + name: value for name, value in weights.items()}
            weights["mtp.embed_tokens.weight"] = embedding
        served.load_weights(weights.items())
        # A missing computation field is a test failure even if SGLang
        # could construct with a fallback value.
        if not native:
            self.assertFalse(
                read._reads - read._written - NON_COMPUTATIONAL_READS,
                read._reads - read._written - NON_COMPUTATIONAL_READS,
            )
        slots = (
            config.spec_config.protocol.num_model_slots
            if config.spec_config.protocol
            else 4
        )
        ids = torch.arange(slots, device=device)[None] % config.vocab_size
        nctx = 0 if eagle or native else 3
        width = (
            config.spec_config.target_features.input_width
            if eagle
            else config.hidden_size if native else config.hidden_size * 2
        )
        features = torch.randn(
            1, slots if eagle or native else nctx, width, device=device
        )
        positions = torch.arange(nctx + slots, device=device)[None]
        layout = AttentionLayout(
            query_positions=positions[:, nctx:],
            key_positions=positions,
            visibility=torch.ones(
                1, 1, slots, nctx + slots, dtype=torch.bool, device=device
            ),
        )
        entries = [
            LayerInputs(attention_layout=layout, position_ids=positions)
        ] * config.num_hidden_layers
        context = {}
        if not eagle and not native:
            projected = served.project_target_hidden(features[0])
            for layer in served.layers:
                attn = layer.self_attn
                k, v = attn.kv_proj_only(projected)
                k = attn.apply_k_rope(positions[0, :nctx], attn.apply_k_norm(k))
                context[attn.attn.layer_id] = (k, v)

        dense = dense_attention(context, positions, nctx, slots)

        with (
            patch.object(RadixAttention, "forward", dense),
            torch.no_grad(),
        ):
            if native:
                layer = served.model.layers[0]
                h = served.fc(
                    torch.cat(
                        [
                            served.pre_fc_norm_embedding(embedding[ids[0]]),
                            served.pre_fc_norm_hidden(features[0]),
                        ],
                        -1,
                    )
                )
                h = h + layer.self_attention(
                    positions[0],
                    layer.input_layernorm(h),
                    SimpleNamespace(),
                )
                h = h + layer.mlp(layer.post_attention_layernorm(h))
                expected = served.model.norm(h)
                actual = model(ids, features, entries)
                torch.testing.assert_close(
                    actual.hidden_states[0], expected, rtol=2e-5, atol=2e-6
                )
            elif eagle:
                batch = SimpleNamespace(
                    spec_info=SimpleNamespace(hidden_states=features[0])
                )
                expected, aux = served.model(
                    ids[0],
                    positions[0],
                    batch,
                    input_embeds=embedding[ids[0]],
                )
                actual = model(ids, features, entries)
                torch.testing.assert_close(
                    actual.hidden_states[0], aux[0], rtol=2e-5, atol=2e-6
                )
                torch.testing.assert_close(
                    actual.logits[0],
                    torch.nn.functional.linear(
                        expected,
                        served.lm_head.weight[: config.draft_vocab_size],
                    ),
                    rtol=2e-5,
                    atol=2e-6,
                )
            else:
                expected = served(
                    ids[0],
                    positions[0, nctx:],
                    SimpleNamespace(),
                    input_embeds=embedding[ids[0]],
                ).hidden_states
                actual = model(ids, features, entries, output_logits=False)
                torch.testing.assert_close(
                    actual.hidden_states[0], expected, rtol=2e-5, atol=2e-6
                )
                self._check_heads(model, served, actual.hidden_states, ids)
                self.assertEqual(
                    served.block_size,
                    config.spec_config.protocol.num_model_slots,
                )

    def _check_heads(self, model, served, hidden, ids):
        with torch.no_grad():
            if model.config.draft.architecture == "DFlash2DraftModel":
                args = dict(
                    candidate_ids=torch.randint(64, (*ids.shape, 3)),
                    unary_logits=torch.randn(*ids.shape, 3),
                    hidden_states=hidden,
                    anchor_token_ids=ids[:, 0],
                )
                torch.testing.assert_close(
                    model.candidate_selector.build_lattice(**args),
                    served.candidate_selector.build_lattice(**args),
                )
                logits = torch.randn(1, 4, 64)
                torch.testing.assert_close(
                    model.transform_unary_logits(logits.clone()),
                    served._transform_unary_logits(logits.clone()),
                )
            if model.config.draft.architecture == "DSparkDraftModel":
                state = served.markov_head.init_state(hidden[:, 0])
                bias = []
                for i in range(ids.shape[1]):
                    value, state = served.markov_head.step_bias(
                        ids[:, i], hidden_states=hidden[:, i], state=state
                    )
                    bias.append(value)
                actual = model.markov_head.apply_block_logits(
                    torch.zeros(*ids.shape, 64), token_ids=ids, hidden_states=hidden
                )
                torch.testing.assert_close(actual, torch.stack(bias, 1))
                previous = model.markov_head.get_prev_embeddings(ids)
                torch.testing.assert_close(
                    model.confidence_head(torch.cat([hidden, previous], -1)),
                    served.confidence_head(hidden, previous),
                )
            if model.config.draft.architecture == "DominoDraftModel":
                prefix = torch.randn(1, hidden.shape[1] + 1, hidden.shape[-1])
                gru, _ = served.prefix_gru(prefix)
                expected = served.embed_proj(torch.cat([hidden, gru[:, 1:]], -1))
                torch.testing.assert_close(model.gru_head(hidden, prefix), expected)

    def test_canonical_hf_remote_code_is_rejected_by_serving(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from sglang.srt.configs.model_config import ModelImpl
        from sglang.srt.model_loader.utils import (
            get_model_architecture,
            resolve_transformers_arch,
        )
        from transformers import AutoConfig

        from specforge.checkpoint import save_draft
        from specforge.export.to_hf import export_to_hf

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_draft(build_draft_model(serving_configs()[0]), root / "draft")
            export_to_hf(str(root / "draft"), None, str(root / "hf"))
            hf = AutoConfig.from_pretrained(root / "hf", trust_remote_code=True)
            config = SimpleNamespace(
                hf_config=hf,
                hf_text_config=hf,
                model_impl=ModelImpl.AUTO,
                model_path=str(root / "hf"),
                revision=None,
                quantization=None,
                is_generation=True,
                is_multimodal=False,
            )
            # A shared class name can select SGLang's native implementation,
            # but the HF shell exposes no serving structural fields.
            model_class, architecture = get_model_architecture(config)
            self.assertEqual(architecture, "DFlashDraftModel")
            with self.assertRaisesRegex(AttributeError, "hidden_size"):
                model_class(hf)
            with self.assertRaisesRegex(ValueError, "not compatible with SGLang"):
                resolve_transformers_arch(config, hf.architectures)
