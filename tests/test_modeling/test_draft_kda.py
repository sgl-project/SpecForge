"""KDA reads each block's visible context as recurrent state, exactly."""

import importlib.util
import unittest
from dataclasses import replace

import torch
import yaml
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.config.draft.attention import KDASpec
from specforge.config.draft.codec import parse_draft
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.rope import DefaultRoPESpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.attention.backends import AttentionPreparationScope
from specforge.modeling.draft.layers.attention.kda import (
    KDAAttention,
    PreparedContextScan,
)
from specforge.modeling.draft.layers.attention.kda.kernels import fla_kda, reference_kda
from specforge.modeling.draft.layers.attention.kda.scan import (
    build_context_scan_plan,
    scan_context_states,
)
from specforge.modeling.draft.layers.attention.masks import (
    AttentionLayout,
    ContextBlocks,
)
from tests.test_algorithms.test_objectives import training_batch
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_kda_spec import PRESET
from tests.test_modeling.test_canonical_dflash2 import dflash2_config
from tests.test_modeling.test_canonical_dmtp import model_config as dmtp_config
from tests.test_modeling.test_canonical_domino import model_config as domino_config
from tests.test_modeling.test_canonical_dspark import model_config as dspark_config

CUDA_AND_FLA = torch.cuda.is_available() and importlib.util.find_spec("fla") is not None
WIDTH, HEADS, HEAD_DIM, BLOCK, CONV = 24, 3, 8, 3, 3


def assert_same_recurrence(actual, expected):
    # CPU exp/sigmoid vectorization rounds by tensor length; splitting a
    # sequence moves results by a few FP32 ulps, never more.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def kda_spec(**overrides):
    values = dict(
        rope_parameters=None,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        short_conv_kernel_size=CONV,
        gate_lower_bound=-5.0,
        full_rank_output_gate=False,
    )
    return KDASpec(**(values | overrides))


def kda_layer(spec=None, *, dtype=torch.float32, device=None):
    torch.manual_seed(0)
    layer = KDAAttention(
        spec or kda_spec(),
        input_size=WIDTH,
        hidden_size=WIDTH,
        norm=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0),
        rope=DefaultRoPESpec(
            interleaved=False, rope_theta=10000.0, partial_rotary_factor=1.0
        ),
        backend="eager",
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.3)
    return layer


def block_layout(lengths, *, rows, active=None, valid=None, block=BLOCK):
    batch, count = lengths.shape
    device = lengths.device
    queries = (lengths[..., None] + torch.arange(block, device=device)).flatten(1)
    keys = torch.arange(rows, device=device).expand(batch, -1)
    return AttentionLayout(
        query_positions=queries,
        key_positions=torch.cat((keys, queries), dim=1),
        context_blocks=ContextBlocks(
            block_size=block,
            context_lengths=lengths,
            context_valid=(
                torch.ones(batch, rows, dtype=torch.bool, device=device)
                if valid is None
                else valid
            ),
            active_blocks=(
                torch.ones_like(lengths, dtype=torch.bool) if active is None else active
            ),
        ),
    )


def run(layer, layout, hidden, context):
    with AttentionPreparationScope() as scope:
        prepared = layer.prepare(layout, scope, batch_size=hidden.shape[0])
    return layer(
        hidden, attention=prepared, position_embeddings=None, target_context=context
    )


def contiguous_oracle(layer, context, block):
    """KDA over the contiguous sequence ``[context ; block]``, block rows only."""
    sequence = torch.cat((context, block), dim=1)
    length = sequence.shape[1]
    shape = (1, length, HEADS, HEAD_DIM)
    q, k, v = (
        conv(proj(sequence)).view(shape)
        for conv, proj in (
            (layer.q_conv1d, layer.q_proj),
            (layer.k_conv1d, layer.k_proj),
            (layer.v_conv1d, layer.v_proj),
        )
    )
    gate = layer.f_b_proj(layer.f_a_proj(sequence)).view(shape)
    beta = layer.b_proj(sequence).view(1, length, HEADS)
    output = reference_kda(
        q, k, v, gate, beta, layer.A_log, layer.dt_bias, layer.config.gate_lower_bound
    )[:, context.shape[1] :]
    output_gate = layer.g_b_proj(layer.g_a_proj(block)).view(
        1, block.shape[1], HEADS, HEAD_DIM
    )
    return layer.o_proj(layer.o_norm(output, output_gate).flatten(-2))


def kda_hybrid(config):
    """Replace the first layer's attention with KDA, keeping its MLP and conv."""
    first = config.layer_configs[config.layer_types[0]]
    return replace(
        config,
        attention_configs=dict(config.attention_configs) | {"kda": kda_spec()},
        layer_configs=dict(config.layer_configs)
        | {"kda": replace(first, attention="kda")},
        layer_types=("kda", *config.layer_types[1:]),
    )


class ReferenceKernelTest(unittest.TestCase):
    def test_state_carries_across_split_and_variable_length_sequences(self):
        torch.manual_seed(0)
        q, k, gate = (torch.randn(1, 9, HEADS, HEAD_DIM) for _ in range(3))
        v, beta = torch.randn(1, 9, HEADS, HEAD_DIM), torch.randn(1, 9, HEADS)
        A_log, dt_bias = torch.rand(HEADS), torch.randn(HEADS * HEAD_DIM)
        whole, state = reference_kda(
            q, k, v, gate, beta, A_log, dt_bias, -5.0, output_final_state=True
        )
        head, middle = reference_kda(
            *(t[:, :4] for t in (q, k, v, gate, beta)),
            A_log,
            dt_bias,
            -5.0,
            output_final_state=True,
        )
        tail, end = reference_kda(
            *(t[:, 4:] for t in (q, k, v, gate, beta)),
            A_log,
            dt_bias,
            -5.0,
            initial_state=middle,
            output_final_state=True,
        )
        assert_same_recurrence(torch.cat((head, tail), dim=1), whole)
        assert_same_recurrence(end, state)
        packed, finals = reference_kda(
            q,
            k,
            v,
            gate,
            beta,
            A_log,
            dt_bias,
            -5.0,
            initial_state=torch.stack((torch.zeros_like(middle[0]), middle[0])),
            output_final_state=True,
            cu_seqlens=torch.tensor([0, 4, 9]),
        )
        assert_same_recurrence(packed, torch.cat((head, tail), dim=1))
        assert_same_recurrence(finals[1], end[0])


class ContextScanTest(unittest.TestCase):
    def test_every_block_starts_from_a_scan_of_its_context_prefix(self):
        generator = torch.Generator().manual_seed(3)
        A_log, dt_bias = torch.rand(HEADS), torch.randn(HEADS * HEAD_DIM)
        for trial in range(12):
            batch, rows = 1 + trial % 3, 3 + 7 * trial
            count = 1 + (5 * trial) % 23
            lengths = torch.randint(0, rows + 1, (batch, count), generator=generator)
            lengths[0, 0] = 0
            lengths[-1, -1] = rows
            if count > 2:
                lengths[0, 1] = lengths[0, 2]
            k, gate = (
                torch.randn(batch, rows, HEADS, HEAD_DIM, generator=generator)
                for _ in range(2)
            )
            v = torch.randn(batch, rows, HEADS, HEAD_DIM, generator=generator)
            beta = torch.randn(batch, rows, HEADS, generator=generator)
            for group_size in (None, 1, 3):
                with self.subTest(trial=trial, group_size=group_size):
                    blocks = block_layout(lengths, rows=rows).context_blocks
                    plan = build_context_scan_plan(blocks, group_size=group_size)
                    states = scan_context_states(
                        reference_kda, k, v, gate, beta, A_log, dt_bias, -5.0, plan
                    )
                    for b in range(batch):
                        for n in range(count):
                            length = int(lengths[b, n])
                            _, expected = reference_kda(
                                torch.zeros_like(k[b : b + 1, :length]),
                                *(t[b : b + 1, :length] for t in (k, v, gate, beta)),
                                A_log,
                                dt_bias,
                                -5.0,
                                output_final_state=True,
                            )
                            assert_same_recurrence(states[b, n], expected[0])

    def test_plan_scans_each_needed_segment_once(self):
        def sequence_lengths(launches):
            return [
                [
                    int(end - start)
                    for start, end in zip(
                        launch.cu_seqlens_cpu[: launch.num_sequences],
                        launch.cu_seqlens_cpu[1 : launch.num_sequences + 1],
                    )
                ]
                for launch in launches
            ]

        # Segments [0,2) [2,5) | [5,9) [9,12): spans chain the two groups, and
        # only segments that do not end a group need their own pass.
        plan = build_context_scan_plan(
            block_layout(torch.tensor([[2, 5, 9, 12]]), rows=12).context_blocks,
            group_size=2,
        )
        self.assertEqual(sequence_lengths(plan.span_launches), [[5], [7]])
        self.assertEqual(sequence_lengths(plan.segment_launches), [[2, 4]])
        # An inactive block reads nothing, whatever length it carries.
        plan = build_context_scan_plan(
            block_layout(
                torch.tensor([[2, 7]]),
                rows=12,
                active=torch.tensor([[True, False]]),
            ).context_blocks
        )
        self.assertEqual(plan.context_lengths.tolist(), [[2, 0]])
        scanned = sequence_lengths(plan.span_launches + plan.segment_launches)
        self.assertEqual(sum(map(sum, scanned)), 2)

    def test_visible_context_must_be_an_unpadded_prefix(self):
        lengths = torch.tensor([[2, 5]])
        valid = torch.ones(1, 6, dtype=torch.bool)
        valid[0, 4] = False
        with self.assertRaisesRegex(ValueError, "unpadded"):
            build_context_scan_plan(
                block_layout(lengths, rows=6, valid=valid).context_blocks
            )
        # an inactive block reads nothing, so padding before its length is fine
        build_context_scan_plan(
            block_layout(
                lengths, rows=6, valid=valid, active=torch.tensor([[True, False]])
            ).context_blocks
        )
        with self.assertRaisesRegex(ValueError, r"\[0, context rows\]"):
            build_context_scan_plan(
                block_layout(torch.tensor([[2, 7]]), rows=6).context_blocks
            )


class KDAAttentionTest(unittest.TestCase):
    def test_blocks_continue_the_recurrence_of_their_visible_context(self):
        for spec in (kda_spec(), kda_spec(short_conv_kernel_size=1)):
            with self.subTest(conv=spec.short_conv_kernel_size):
                layer = kda_layer(spec)
                rows = 11
                lengths = torch.tensor([[0, 4, 11], [7, 7, 2]])
                hidden = torch.randn(2, 3 * BLOCK, WIDTH)
                context = torch.randn(2, rows, WIDTH)
                output = run(layer, block_layout(lengths, rows=rows), hidden, context)
                for b in range(2):
                    for n in range(3):
                        length = int(lengths[b, n])
                        expected = contiguous_oracle(
                            layer,
                            context[b : b + 1, :length],
                            hidden[b : b + 1, n * BLOCK : (n + 1) * BLOCK],
                        )
                        torch.testing.assert_close(
                            output[b : b + 1, n * BLOCK : (n + 1) * BLOCK], expected
                        )

    def test_inactive_blocks_return_zero(self):
        layer = kda_layer()
        active = torch.tensor([[True, False, True]])
        output = run(
            layer,
            block_layout(torch.tensor([[1, 3, 5]]), rows=6, active=active),
            torch.randn(1, 3 * BLOCK, WIDTH),
            torch.randn(1, 6, WIDTH),
        )
        self.assertTrue(torch.all(output[:, BLOCK : 2 * BLOCK] == 0))
        self.assertTrue(torch.all(output[:, :BLOCK] != 0))

    def test_layers_of_one_forward_share_one_scan_plan(self):
        first, second = kda_layer(), kda_layer()
        layout = block_layout(torch.tensor([[1, 3]]), rows=4)
        with AttentionPreparationScope() as scope:
            a = first.prepare(layout, scope, batch_size=1)
            b = second.prepare(layout, scope, batch_size=1)
        self.assertIsInstance(a, PreparedContextScan)
        self.assertIs(a, b)
        with AttentionPreparationScope() as scope:
            self.assertIsNot(first.prepare(layout, scope, batch_size=1), a)

    def test_layouts_without_context_blocks_and_bad_inputs_are_rejected(self):
        layer = kda_layer()
        dense = AttentionLayout(
            query_positions=torch.arange(BLOCK)[None],
            key_positions=torch.arange(BLOCK)[None],
            visibility=torch.ones(1, 1, BLOCK, BLOCK, dtype=torch.bool),
        )
        with AttentionPreparationScope() as scope:
            with self.assertRaisesRegex(ValueError, "context_blocks"):
                layer.prepare(dense, scope, batch_size=1)
            prepared = layer.prepare(
                block_layout(torch.tensor([[2]]), rows=4), scope, batch_size=1
            )
        hidden, context = torch.randn(1, BLOCK, WIDTH), torch.randn(1, 4, WIDTH)
        with self.assertRaisesRegex(ValueError, "rotary"):
            layer(
                hidden,
                attention=prepared,
                position_embeddings=(context, context),
                target_context=context,
            )
        with self.assertRaisesRegex(ValueError, "target_context"):
            layer(hidden, attention=prepared, position_embeddings=None)

    def test_initialization_draws_kimi_decay_and_respects_loaded_A_log(self):
        layer = KDAAttention(
            kda_spec(full_rank_output_gate=True),
            input_size=WIDTH,
            hidden_size=WIDTH,
            norm=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0),
            rope=DefaultRoPESpec(
                interleaved=False, rope_theta=10000.0, partial_rotary_factor=1.0
            ),
            backend="eager",
        )
        decay = layer.A_log.exp()
        self.assertTrue(torch.all((decay >= 1) & (decay <= 16)))
        self.assertTrue(torch.all(layer.dt_bias == 0))
        self.assertTrue(torch.all(layer.o_norm.weight == 1))
        self.assertTrue(hasattr(layer, "g_proj") and not hasattr(layer, "g_a_proj"))
        with torch.no_grad():
            layer.A_log.fill_(0.5)
        layer.A_log._is_hf_initialized = True
        layer.reset_draft_parameters()
        self.assertTrue(torch.all(layer.A_log == 0.5))


class KDAHybridTrainingTest(unittest.TestCase):
    def test_preset_builds_kimi_shaped_kda_layers(self):
        config = parse_draft(yaml.safe_load(PRESET.read_text()), target_config=None)
        model = build_draft_model(config, device="meta")
        kinds = [type(layer.self_attn) for layer in model.backbone.layers]
        self.assertEqual(kinds.count(KDAAttention), 4)
        attention = model.backbone.layers[0].self_attn
        self.assertEqual(attention.q_proj.weight.shape, (2048, 5120))
        self.assertEqual(attention.A_log.shape, (16,))

    def test_hybrid_stacks_train_every_parameter_on_block_architectures(self):
        for config in (
            draft_config(target_bindings=(), target_logit_transform=None),
            dflash2_config(),
            dspark_config(),
            domino_config(),
            dmtp_config(),
        ):
            config = kda_hybrid(config)
            with self.subTest(architecture=config.architecture):
                torch.manual_seed(11)
                model = build_draft_model(config)
                self.assertIsInstance(model.backbone.layers[0].self_attn, KDAAttention)
                for name in config.target_bindings:
                    model.get_submodule(name).load_target_weight(
                        torch.randn(config.vocab_size, config.hidden_size)
                    )
                runtime = build_objective_model(
                    model,
                    options={"num_anchors": 2},
                    target_head=nn.Linear(32, config.vocab_size, bias=False),
                )
                result = runtime(training_batch(config), step=1, total_steps=10)
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


@unittest.skipUnless(CUDA_AND_FLA, "requires CUDA and fla-core")
class FLAKernelTest(unittest.TestCase):
    def test_fla_matches_the_reference_with_states_and_variable_lengths(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        q, k, gate = (
            torch.randn(1, 50, HEADS, HEAD_DIM, device=device) for _ in range(3)
        )
        v = torch.randn(1, 50, HEADS, HEAD_DIM, device=device)
        beta = torch.randn(1, 50, HEADS, device=device)
        A_log = torch.rand(HEADS, device=device)
        dt_bias = torch.randn(HEADS * HEAD_DIM, device=device)
        states = torch.randn(3, HEADS, HEAD_DIM, HEAD_DIM, device=device) * 0.1
        cu_seqlens = torch.tensor([0, 7, 30, 50], device=device)
        for lower_bound in (-5.0, None):
            with self.subTest(lower_bound=lower_bound):
                expected = reference_kda(
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    A_log,
                    dt_bias,
                    lower_bound,
                    initial_state=states,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                )
                actual = fla_kda(
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    A_log,
                    dt_bias,
                    lower_bound,
                    initial_state=states,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                )
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, rtol=2e-3, atol=2e-3)

    def test_cuda_layer_matches_the_cpu_reference_layer(self):
        layer = kda_layer()
        lengths = torch.tensor([[0, 5, 9, 40], [3, 3, 60, 17]])
        hidden = torch.randn(2, 4 * BLOCK, WIDTH)
        context = torch.randn(2, 60, WIDTH)
        expected = run(layer, block_layout(lengths, rows=60), hidden, context)
        cuda = layer.cuda()
        actual = run(
            cuda,
            block_layout(lengths.cuda(), rows=60),
            hidden.cuda(),
            context.cuda(),
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)
