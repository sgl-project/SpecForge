"""The HF capture backend's Qwen3.8-Flash-Next pieces against the Transformers reference."""

import copy
import tempfile
import unittest

import torch

try:
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
except ImportError:  # transformers < 5.16
    Qwen4ExpConfig = None

from specforge.offline_capture import qwen4_exp
from specforge.offline_capture.hf_backend import (
    OfflineHFCaptureBackend,
    dispatch_layers,
    plan_layer_devices,
)

TAPS = [2, 3, 5]


def tiny_config(indexer_budget=16):
    """Flash-Next's layout at toy width: 3 GDN + 1 QSA, PLE before layer 1."""

    text = dict(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=8,
        layer_types=["linear_attention"] * 3
        + ["full_attention"]
        + ["linear_attention"] * 3
        + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        partial_rotary_factor=0.25,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=1000,
        split_ngram_parts=4,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=indexer_budget,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        eos_token_id=500,
        bos_token_id=500,
        max_position_embeddings=4096,
    )
    vision = dict(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        deepstack_visual_indexes=[],
    )
    return Qwen4ExpConfig(
        text_config=text,
        vision_config=vision,
        architectures=["Qwen4ExpForConditionalGeneration"],
    )


def tiny_model(attn_implementation="sdpa", seed=0):
    torch.manual_seed(seed)
    config = tiny_config()
    config._attn_implementation = attn_implementation
    config.text_config._attn_implementation = attn_implementation
    model = Qwen4ExpForConditionalGeneration(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() >= 2:
                parameter.normal_(0.0, 0.2)
    return model.requires_grad_(False)


def tokens(length, seed=1):
    generator = torch.Generator().manual_seed(seed)
    # 500 is EOS: keep it out so the n-gram history is not reset.
    return torch.randint(0, 499, (1, length), generator=generator)


@unittest.skipIf(Qwen4ExpConfig is None, "transformers has no qwen4_exp")
class TestFastPaths(unittest.TestCase):
    def _forward(self, model, input_ids):
        out = qwen4_exp.text_model(model)(
            input_ids=input_ids, use_cache=False, output_hidden_states=True
        )
        return out.last_hidden_state, out.hidden_states

    def test_indexer_masks_match_the_reference(self):
        # Budget 16 keeps 4 blocks: queries past position 18 are truly sparse.
        for implementation in ("sdpa", "eager"):
            model = tiny_model(implementation)
            text = qwen4_exp.text_model(model)
            sparse = [
                layer.self_attn.indexer
                for layer in text.layers
                if hasattr(layer, "self_attn")
            ]
            captured = []

            def keep_inputs(module, args, kwargs):
                captured.append((module, args, kwargs))

            handles = [
                m.register_forward_pre_hook(keep_inputs, with_kwargs=True)
                for m in sparse
            ]
            self._forward(model, tokens(70))
            for handle in handles:
                handle.remove()
            self.assertEqual(len(captured), 2)
            for module, args, kwargs in captured:
                reference = module(*args, **kwargs)
                fast = qwen4_exp.fast_indexer_forward(module, *args, **kwargs)
                self.assertEqual(fast.dtype, reference.dtype)
                self.assertTrue(torch.equal(fast, reference), implementation)

    def test_indexer_is_dense_within_the_budget(self):
        model = tiny_model()
        text = qwen4_exp.text_model(model)
        module = text.layers[3].self_attn.indexer
        captured = []
        handle = module.register_forward_pre_hook(
            lambda m, a, k: captured.append((a, k)), with_kwargs=True
        )
        # 16-token budget plus the 3-token tail: every query sees everything.
        self._forward(model, tokens(19))
        handle.remove()
        args, kwargs = captured[0]
        mask = qwen4_exp.fast_indexer_forward(module, *args, **kwargs)
        causal = torch.ones(19, 19, dtype=torch.bool).tril()
        self.assertTrue(torch.equal(mask[0, 0] & causal, causal))

    def test_sorted_experts_match_the_reference(self):
        model = tiny_model()
        experts = qwen4_exp.text_model(model).layers[0].mlp.experts
        generator = torch.Generator().manual_seed(3)
        hidden = torch.randn(37, 64, generator=generator)
        index = torch.stack(
            [torch.randperm(8, generator=generator)[:2] for _ in range(37)]
        )
        weights = torch.rand(37, 2, generator=generator)
        reference = experts(hidden, index, weights)
        fast = qwen4_exp.sorted_experts_forward(experts, hidden, index, weights)
        self.assertTrue(torch.allclose(fast, reference, atol=1e-6, rtol=0))

    def test_installed_fast_paths_keep_the_forward(self):
        reference = tiny_model()
        fast = copy.deepcopy(reference)
        counts = qwen4_exp.install_fast_paths(fast)
        self.assertEqual(counts, {"indexer": 2, "experts": 8})
        ids = tokens(70)
        expected, _ = self._forward(reference, ids)
        actual, _ = self._forward(fast, ids)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=0))


@unittest.skipIf(Qwen4ExpConfig is None, "transformers has no qwen4_exp")
class TestTaps(unittest.TestCase):
    def test_tap_is_the_next_layers_gated_read(self):
        model = tiny_model()
        text = qwen4_exp.text_model(model)
        sink = {}
        handles = qwen4_exp.hyper_connection_taps(model, TAPS, sink)
        out = text(input_ids=tokens(30), use_cache=False, output_hidden_states=True)
        for handle in handles:
            handle.remove()
        for layer_id in TAPS:
            # hidden_states[i + 1] is layer i's output: the 4-stream residual.
            streams = out.hidden_states[layer_id + 1]
            self.assertEqual(streams.shape[-1], 4 * 64)
            expected = text.layers[layer_id + 1].attn_hyper_connection(streams)[0]
            self.assertEqual(sink[layer_id].shape, (1, 30, 64))
            self.assertTrue(torch.equal(sink[layer_id], expected))

    def test_taps_without_a_plain_next_layer_are_refused(self):
        model = tiny_model()
        with self.assertRaisesRegex(ValueError, "PLE"):
            qwen4_exp.hyper_connection_taps(model, [0], {})
        with self.assertRaisesRegex(ValueError, "no following layer"):
            qwen4_exp.hyper_connection_taps(model, [7], {})


class TestPlacement(unittest.TestCase):
    def test_layers_split_by_bytes_in_order(self):
        plan = plan_layer_devices([10] * 48, 8, head_bytes=0, tail_bytes=0)
        self.assertEqual(plan, sorted(plan))
        self.assertEqual([plan.count(d) for d in range(8)], [6] * 8)

    def test_a_heavy_head_pushes_layers_off_the_first_device(self):
        plan = plan_layer_devices([10] * 8, 2, head_bytes=40, tail_bytes=0)
        self.assertEqual(plan.count(0), 2)

    def test_one_device_takes_everything(self):
        self.assertEqual(plan_layer_devices([5, 5, 5], 1, 0, 0), [0, 0, 0])


@unittest.skipIf(Qwen4ExpConfig is None, "transformers has no qwen4_exp")
class TestDispatch(unittest.TestCase):
    """Dispatch onto the meta device, so a CPU-only host can see a migration.

    On CPU every placement is a no-op. Meta is a real second device, and a
    tensor sent there cannot come back, so a table that was moved even
    transiently would lose its storage.
    """

    def test_n_gram_table_is_never_migrated(self):
        model = tiny_model()
        ple = qwen4_exp.text_model(model).layers[1].ple
        table = ple.ple_embedding.ngram_embedding.weight
        storage = table.data_ptr()

        dispatch_layers(model, [torch.device("meta")] * 2)

        self.assertIs(ple.ple_embedding.ngram_embedding.weight, table)
        self.assertEqual(table.device.type, "cpu")
        self.assertEqual(table.data_ptr(), storage)
        # The rest of the PLE layer still runs on its layer's device.
        self.assertEqual(ple.key_proj.weight.device.type, "meta")
        self.assertEqual(ple.conv1d.weight.device.type, "meta")
        self.assertEqual(ple.ple_embedding.layer_multipliers.device.type, "meta")
        self.assertEqual(ple.ple_embedding.ngram_heads_offsets.device.type, "meta")

    def test_everything_else_is_placed(self):
        model = tiny_model()
        table = qwen4_exp.text_model(model).layers[1].ple.ple_embedding.ngram_embedding
        dispatch_layers(model, [torch.device("meta")] * 2)
        host = {id(tensor) for tensor in table.parameters()}
        stray = [
            name
            for name, tensor in list(model.named_parameters())
            + list(model.named_buffers())
            if id(tensor) not in host and tensor.device.type != "meta"
        ]
        self.assertEqual(stray, [])

    def test_a_table_loaded_off_host_is_refused(self):
        model = tiny_model()
        table = qwen4_exp.text_model(model).layers[1].ple.ple_embedding.ngram_embedding
        table.to("meta")
        with self.assertRaisesRegex(RuntimeError, "host-resident"):
            dispatch_layers(model, [torch.device("cpu")])


@unittest.skipIf(Qwen4ExpConfig is None, "transformers has no qwen4_exp")
class TestBackend(unittest.TestCase):
    def test_capture_rows_match_the_reference_forward(self):
        reference = tiny_model()
        with tempfile.TemporaryDirectory() as path:
            reference.save_pretrained(path)
            backend = OfflineHFCaptureBackend.build(path, torch_dtype=torch.float32)
        backend.set_capture_layers(TAPS, capture_method="dspark")
        # The backend runs the fast paths; give the reference the same ones so
        # this checks the capture alone, bit for bit. Against the stock forward
        # the rows differ wherever the indexer's top-k boundary falls on tied
        # scores (ReLU zeros: row 0, layer 3, query 21 here) -- which tied block
        # torch.topk keeps is platform-dependent (aarch64 differs from x86/macOS).
        # TestFastPaths checks the fast paths against the stock forward.
        qwen4_exp.install_fast_paths(reference)

        ids = torch.cat([tokens(40, seed=5), tokens(40, seed=6)])
        mask = torch.ones_like(ids)
        loss = torch.zeros_like(ids)
        data, aux_rows, last_rows = backend.capture(
            input_ids=ids, attention_mask=mask, loss_mask=loss
        )
        self.assertEqual(len(data), 2)
        text = qwen4_exp.text_model(reference)
        for row in range(2):
            self.assertEqual(aux_rows[row].shape, (40, len(TAPS) * 64))
            out = text(
                input_ids=ids[row : row + 1],
                use_cache=False,
                output_hidden_states=True,
            )
            expected = torch.cat(
                [
                    text.layers[i + 1].attn_hyper_connection(out.hidden_states[i + 1])[
                        0
                    ]
                    for i in TAPS
                ],
                dim=-1,
            )[0]
            self.assertTrue(torch.equal(aux_rows[row], expected))
            self.assertTrue(torch.equal(last_rows[row], out.last_hidden_state[0]))
            self.assertTrue(torch.equal(data[row][0], ids[row : row + 1]))

    def test_padded_rows_are_captured_over_their_real_tokens(self):
        reference = tiny_model()
        with tempfile.TemporaryDirectory() as path:
            reference.save_pretrained(path)
            backend = OfflineHFCaptureBackend.build(path, torch_dtype=torch.float32)
        backend.set_capture_layers(TAPS, capture_method="dspark")
        short = tokens(24, seed=7)
        ids = torch.cat([tokens(40, seed=5), torch.nn.functional.pad(short, (0, 16))])
        mask = torch.ones_like(ids)
        mask[1, 24:] = 0
        _, aux_rows, last_rows = backend.capture(
            input_ids=ids, attention_mask=mask, loss_mask=torch.zeros_like(ids)
        )
        alone_aux, alone_last = backend._capture_row(short)
        self.assertEqual(aux_rows[1].shape, (40, len(TAPS) * 64))
        self.assertTrue(torch.equal(aux_rows[1][:24], alone_aux))
        self.assertTrue(torch.equal(last_rows[1][:24], alone_last))
        self.assertFalse(aux_rows[1][24:].any())
        self.assertFalse(last_rows[1][24:].any())

    def test_n_gram_table_stays_in_host_memory(self):
        reference = tiny_model()
        with tempfile.TemporaryDirectory() as path:
            reference.save_pretrained(path)
            backend = OfflineHFCaptureBackend.build(path, torch_dtype=torch.float32)
        table = (
            qwen4_exp.text_model(backend.model)
            .layers[1]
            .ple.ple_embedding.ngram_embedding
        )
        self.assertEqual(table.weight.device.type, "cpu")
        self.assertNotEqual(table.weight.device.type, "meta")

    def test_loader_wraps_the_backend_like_sglang(self):
        from specforge.offline_capture import load_offline_capture

        reference = tiny_model()
        with tempfile.TemporaryDirectory() as path:
            reference.save_pretrained(path)
            capture = load_offline_capture(
                path, torch_dtype=torch.float32, backend="hf"
            )
        capture.set_capture_layers(TAPS, capture_method="dspark")
        ids = tokens(24)
        batch = capture.capture(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            loss_mask=torch.ones_like(ids),
        )
        self.assertEqual(batch.hidden_states.shape, (1, 24, len(TAPS) * 64))
        self.assertEqual(batch.last_hidden_states.shape, (1, 24, 64))
        self.assertTrue(torch.equal(batch.input_ids, ids))

    def test_only_dflash_family_capture_is_offered(self):
        reference = tiny_model()
        with tempfile.TemporaryDirectory() as path:
            reference.save_pretrained(path)
            backend = OfflineHFCaptureBackend.build(path, torch_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "DFlash-family"):
            backend.set_capture_layers(TAPS, capture_method="eagle3")


class TestUnpaddedRows(unittest.TestCase):
    """Row handling around the forward; needs no qwen4_exp."""

    def backend(self):
        backend = OfflineHFCaptureBackend.__new__(OfflineHFCaptureBackend)
        backend.layer_ids = TAPS
        backend.output_device = torch.device("cpu")
        seen = []

        def capture_row(input_ids):
            seen.append(input_ids.shape[1])
            length = input_ids.shape[1]
            return torch.ones(length, 6), torch.ones(length, 2)

        backend._capture_row = capture_row
        return backend, seen

    def test_each_row_runs_over_its_real_tokens_only(self):
        backend, seen = self.backend()
        ids = torch.arange(10).view(2, 5)
        mask = torch.tensor([[1] * 5, [1, 1, 1, 0, 0]])
        _, aux_rows, last_rows = backend.capture(
            input_ids=ids, attention_mask=mask, loss_mask=mask
        )
        self.assertEqual(seen, [5, 3])
        self.assertEqual([tuple(row.shape) for row in aux_rows], [(5, 6), (5, 6)])
        self.assertTrue(torch.equal(aux_rows[1][:, 0], torch.tensor([1.0, 1, 1, 0, 0])))
        self.assertTrue(
            torch.equal(last_rows[1][:, 0], torch.tensor([1.0, 1, 1, 0, 0]))
        )

    def test_the_wrapper_saves_records_at_their_real_length(self):
        from specforge.offline_capture.sglang import OfflineSGLangCapture

        backend, _ = self.backend()
        self.assertTrue(OfflineSGLangCapture(backend).unpadded_rows)

    def test_generate_saves_compact_records_at_their_real_length(self):
        import os
        from pathlib import Path
        from unittest import mock

        from scripts.prepare_hidden_states import HiddenStatesGenerator
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.offline_capture.sglang import OfflineSGLangCapture

        backend, seen = self.backend()
        generator = HiddenStatesGenerator.__new__(HiddenStatesGenerator)
        generator.model = OfflineSGLangCapture(backend)
        generator.capture_layout = (
            builtin_algorithm_registry()
            .resolve("dspark")
            .providers.offline_for("text")
            .capture_layout
        )
        generator.show_progress = False
        generator.file_group_size = 2000
        generator.compress = False
        generator.compression_level = 1
        generator.file_extension = ".ckpt"
        generator.num_io_threads = 2
        generator.pending_futures = []
        generator._save_tensor_async = generator._save_tensor_sync
        mask = torch.zeros(2, 64, dtype=torch.long)
        mask[0] = 1
        mask[1, :3] = 1
        batch = {
            "input_ids": torch.arange(128).view(2, 64),
            "attention_mask": mask,
            "loss_mask": mask,
        }
        fake_dist = mock.Mock()
        fake_dist.get_process_group_ranks.return_value = [0]
        fake_dist.get_rank.return_value = 0
        module = "scripts.prepare_hidden_states"
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch(f"{module}.dist", fake_dist),
                mock.patch(f"{module}.get_tp_group"),
                mock.patch(f"{module}.get_dp_group"),
                mock.patch(f"{module}.is_tp_rank_0", return_value=True),
                mock.patch(f"{module}.get_device_type", return_value="cpu"),
            ):
                generator.generate([batch], tmp, start_idx=0, samples_per_dp=2)
            files = sorted(Path(tmp).rglob("*.ckpt"))
            records = [torch.load(path, weights_only=True) for path in files]
            sizes = [os.path.getsize(path) for path in files]
        self.assertEqual(seen, [64, 3])
        short = min(records, key=lambda record: record["input_ids"].numel())
        self.assertEqual(short["input_ids"].tolist(), [64, 65, 66])
        self.assertEqual(tuple(short["hidden_states"].shape), (1, 3, 6))
        # The saved tensors own only the kept rows: a view of the padded batch
        # row would serialize all 64 rows of its storage.
        for name in ("hidden_states", "target_last_hidden_states"):
            tensor = short[name]
            self.assertEqual(
                tensor.untyped_storage().nbytes(),
                tensor.numel() * tensor.element_size(),
                name,
            )
        self.assertLess(min(sizes), max(sizes) / 2)

    def test_left_padding_is_refused(self):
        backend, _ = self.backend()
        with self.assertRaisesRegex(ValueError, "right-padded"):
            backend.capture(
                input_ids=torch.zeros(1, 4, dtype=torch.long),
                attention_mask=torch.tensor([[0, 1, 1, 1]]),
                loss_mask=torch.zeros(1, 4, dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
