"""DFlash context packing preserves per-document sampling and objectives."""

import copy
import unittest

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.packed_dflash import PackedDFlashLayout


class PackedDFlashMaskTest(unittest.TestCase):
    def test_sparse_tile_metadata_covers_exact_mask_and_full_tiles_are_exact(self):
        lengths = (131, 67, 3, 312)
        starts = torch.tensor([0, 131, 198, 201])
        local = torch.arange(9)[None, :].expand(4, -1) * 13
        local = torch.minimum(local, torch.tensor(lengths)[:, None] - 1)
        anchors = (local + starts[:, None]).reshape(1, -1)
        context_starts = starts[:, None].expand(-1, 9).reshape(1, -1)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        keep[:, 10:13] = False
        for proposal in (3, 16, 33):
            for window in (None, 7):
                for tile_shape in ((128, 128), (256, 128)):
                    with self.subTest(
                        proposal=proposal, window=window, tile_shape=tile_shape
                    ):
                        arguments = dict(
                            anchor_positions=anchors,
                            block_keep_mask=keep,
                            S=sum(lengths),
                            block_size=proposal,
                            device="cpu",
                            sliding_window=window,
                            context_start_positions=context_starts,
                        )
                        dense = create_dflash_sdpa_mask(**arguments)[0, 0]
                        sparse = create_dflash_block_mask(
                            **arguments, flex_block_size=tile_shape
                        )
                        rows, columns = sparse.kv_indices.shape[-2:]
                        tiles = torch.zeros(rows, columns, dtype=torch.bool)
                        full = torch.zeros_like(tiles)
                        for row in range(rows):
                            tiles[
                                row,
                                sparse.kv_indices[
                                    0, 0, row, : sparse.kv_num_blocks[0, 0, row]
                                ],
                            ] = True
                            full[
                                row,
                                sparse.full_kv_indices[
                                    0, 0, row, : sparse.full_kv_num_blocks[0, 0, row]
                                ],
                            ] = True
                        padded = torch.zeros(
                            rows * tile_shape[0],
                            columns * tile_shape[1],
                            dtype=torch.bool,
                        )
                        padded[: dense.shape[0], : dense.shape[1]] = dense
                        token_tiles = padded.reshape(
                            rows, tile_shape[0], columns, tile_shape[1]
                        ).permute(0, 2, 1, 3)
                        exact_any = token_tiles.any(-1).any(-1)
                        exact_full = token_tiles.all(-1).all(-1)
                        self.assertFalse(bool((exact_any & ~(tiles | full)).any()))
                        self.assertFalse(bool((full & ~exact_full).any()))

    def test_sampling_mask_restores_documents_and_excludes_padding(self):
        layout = PackedDFlashLayout.from_lengths((3, 1, 2), 6, "cpu")
        packed_mask = torch.tensor([[1, 0, 1, 1, 1, 1]])
        torch.testing.assert_close(
            layout.padded_loss_mask(packed_mask),
            torch.tensor([[1, 0, 1], [1, 0, 0], [1, 1, 0]]),
        )

    def test_full_and_sliding_masks_never_read_previous_documents(self):
        anchors = torch.tensor([[1, 4, 6]])
        starts = torch.tensor([[0, 3, 3]])
        keep = torch.tensor([[True, False, True]])
        for window in (None, 3):
            args = dict(
                anchor_positions=anchors,
                block_keep_mask=keep,
                S=8,
                block_size=4,
                device="cpu",
                sliding_window=window,
                context_start_positions=starts,
            )
            dense = create_dflash_sdpa_mask(**args)[0, 0]
            sparse = create_dflash_block_mask(**args)
            q = torch.arange(12)[:, None]
            k = torch.arange(20)[None, :]
            actual = sparse.mask_mod(torch.tensor(0), torch.tensor(0), q, k)
            torch.testing.assert_close(actual, dense)
            self.assertFalse(bool(dense[8:, :3].any()))
            self.assertFalse(bool(dense[4:8].any()))
            self.assertFalse(bool(dense[8:, 8:16].any()))


@unittest.skipUnless(
    torch.cuda.is_available(), "production Flex Attention requires CUDA"
)
class PackedDFlashProductionTest(unittest.TestCase):
    def _fixtures(self, *, dflash2, sliding, dtype, loss_type, lk_loss_type=None):
        from transformers import Qwen3Config

        from specforge.modeling.draft.dflash import DFlashDraftModel
        from specforge.modeling.draft.dflash2 import DFlash2DraftModel

        torch.manual_seed(912)
        method = {
            "block_size": 4,
            "mask_token_id": 63,
            "target_layer_ids": [1, 2],
        }
        if dflash2:
            method.update(
                conv_group_size=4, conv_kernel_size=3, selector_rank=8, selector_top_k=8
            )
        config = Qwen3Config(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            num_target_layers=4,
            max_position_embeddings=256,
            vocab_size=64,
            layer_types=(
                ["sliding_attention", "full_attention"]
                if sliding
                else ["full_attention"] * 2
            ),
            sliding_window=16 if sliding else None,
            use_sliding_window=sliding,
            dflash_config=method,
        )
        config._attn_implementation = "flex_attention"
        draft = (
            (DFlash2DraftModel if dflash2 else DFlashDraftModel)(config)
            .cuda()
            .to(dtype)
        )
        if dflash2:
            with torch.no_grad():
                for layer in draft.layers:
                    layer.attention_conv.kernel_projection.weight.normal_(std=0.01)
                    layer.mlp_conv.kernel_projection.weight.normal_(std=0.01)
        head = nn.Linear(64, 64, bias=False).cuda().to(dtype).requires_grad_(False)
        embedding = nn.Embedding(64, 64).cuda().to(dtype).requires_grad_(False)
        model = OnlineDFlashModel(
            draft,
            head,
            embedding,
            mask_token_id=63,
            block_size=4,
            num_anchors=8,
            attention_backend="flex_attention",
            loss_type=loss_type,
            lk_loss_type=lk_loss_type,
            objective_chunk_blocks=3,
            loss_decay_gamma=3.0,
            selector_loss_alpha=0.7,
            metric_top_k=8,
        )
        generator = torch.Generator().manual_seed(555)
        features = []
        for length in (131, 67, 3, 1):
            loss_mask = torch.ones(1, length)
            if length > 10:
                loss_mask[:, :4] = 0
                loss_mask[:, 9:12] = 0
            features.append(
                {
                    "input_ids": torch.randint(0, 63, (1, length), generator=generator),
                    "loss_mask": loss_mask,
                    "hidden_states": torch.randn(
                        1, length, 128, generator=generator
                    ).to(dtype),
                    "target_last_hidden_states": torch.randn(
                        1, length, 64, generator=generator
                    ).to(dtype),
                }
            )
        return model, features

    def _run(self, model, features, *, packed, compact=True):
        lengths = [f["input_ids"].shape[1] for f in features]
        if packed:
            batch = {
                key: torch.cat([f[key] for f in features], dim=1).cuda()
                for key in features[0]
            }
            batch["sequence_lengths"] = tuple(lengths)
            if compact:
                batch["valid_anchor_counts"] = tuple(
                    int(
                        (
                            (f["loss_mask"][:, :-1] > 0.5)
                            & (f["loss_mask"][:, 1:] > 0.5)
                        ).sum()
                    )
                    for f in features
                )
        else:
            batch = {}
            for key in features[0]:
                rows = []
                for f, length in zip(features, lengths):
                    value = f[key]
                    rows.append(
                        torch.cat(
                            (
                                value,
                                value.new_zeros(
                                    (1, max(lengths) - length, *value.shape[2:])
                                ),
                            ),
                            dim=1,
                        )
                    )
                batch[key] = torch.cat(rows).cuda()
        sampled = []
        original_sampler = model._sample_anchor_positions

        def record(*args, **kwargs):
            result = original_sampler(*args, **kwargs)
            sampled.append(tuple(t.detach().clone() for t in result))
            return result

        model._sample_anchor_positions = record
        torch.manual_seed(999)
        try:
            loss, accuracy, metrics = model(**batch)
            loss.backward()
        finally:
            model._sample_anchor_positions = original_sampler
        gradients = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        return (loss.detach(), accuracy.detach(), metrics), gradients, sampled[0]

    def _compare(self, left, right, tol, path=""):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for name in left:
                self._compare(left[name], right[name], tol, path + "/" + name)
        elif isinstance(left, (tuple, list)):
            for index, (a, b) in enumerate(zip(left, right)):
                self._compare(a, b, tol, path + f"/{index}")
        elif isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, msg=path, **tol)
        else:
            self.assertEqual(left, right, path)

    def test_production_anchors_losses_metrics_and_every_gradient_match(self):
        cases = (
            (False, False, torch.float32, "dflash", None),
            (False, True, torch.float32, "dpace", None),
            (True, False, torch.float32, "dflash", None),
            (True, True, torch.float32, "dpace", None),
            (True, False, torch.bfloat16, "dflash", None),
            (True, True, torch.bfloat16, "dpace", None),
            (True, True, torch.float32, "dpace-cumulative-confidence-only", "lambda"),
            (True, False, torch.float32, "dpace-continuation-value-only", "tv"),
        )
        for dflash2, sliding, dtype, loss_type, lk_loss_type in cases:
            with self.subTest(
                dflash2=dflash2,
                sliding=sliding,
                dtype=dtype,
                loss_type=loss_type,
                lk_loss_type=lk_loss_type,
            ):
                model, features = self._fixtures(
                    dflash2=dflash2,
                    sliding=sliding,
                    dtype=dtype,
                    loss_type=loss_type,
                    lk_loss_type=lk_loss_type,
                )
                packed_model = copy.deepcopy(model)
                baseline, baseline_gradients, baseline_anchors = self._run(
                    model, features, packed=False
                )
                packed, packed_gradients, packed_anchors = self._run(
                    packed_model, features, packed=True
                )
                for expected, actual in zip(baseline_anchors, packed_anchors):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                tol = (
                    dict(atol=3e-6, rtol=5e-4)
                    if dtype == torch.float32
                    else dict(atol=7e-4, rtol=5e-2)
                )
                self._compare(packed, baseline, tol)
                self._compare(packed_gradients, baseline_gradients, tol, "gradients")
                self.assertTrue(any("q_proj" in name for name in packed_gradients))
                if dflash2:
                    self.assertTrue(
                        any("candidate_selector" in name for name in packed_gradients)
                    )
                    self.assertTrue(
                        any("attention_conv" in name for name in packed_gradients)
                    )

    def test_dflash2_blocks_cannot_read_another_document(self):
        model, features = self._fixtures(
            dflash2=True, sliding=True, dtype=torch.float32, loss_type="dpace"
        )
        changed = copy.deepcopy(features)
        changed[0]["input_ids"].fill_(33)
        changed[0]["hidden_states"].mul_(100)
        captured = []
        handle = model.draft_model.register_forward_hook(
            lambda _module, _args, output: captured.append(output.detach().clone())
        )
        try:
            self._run(model, features, packed=True)
            model.zero_grad(set_to_none=True)
            self._run(model, changed, packed=True)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 2)
        first_document_blocks = model.num_anchors * model.block_size
        torch.testing.assert_close(
            captured[0][:, first_document_blocks:],
            captured[1][:, first_document_blocks:],
            atol=0,
            rtol=0,
        )

    def test_missing_host_counts_retains_equivalent_uncompacted_fallback(self):
        model, features = self._fixtures(
            dflash2=True, sliding=True, dtype=torch.float32, loss_type="dpace"
        )
        uncompacted_model = copy.deepcopy(model)
        compact, compact_grads, _ = self._run(model, features, packed=True)
        uncompacted, uncompacted_grads, _ = self._run(
            uncompacted_model, features, packed=True, compact=False
        )
        tolerance = dict(atol=3e-6, rtol=5e-4)
        self._compare(compact, uncompacted, tolerance)
        self._compare(compact_grads, uncompacted_grads, tolerance, "gradients")


if __name__ == "__main__":
    unittest.main()
