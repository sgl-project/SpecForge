from __future__ import annotations

import unittest
from types import SimpleNamespace

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.vlm_input import VlmServerInputAdapter
from specforge.algorithms.contracts import FeatureMode
from specforge.data.vlm_preprocessing import _expand_image_region, _image_token_count

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch-free dev boxes
    torch = None
    TORCH_AVAILABLE = False


class MultimodalRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.registry = builtin_algorithm_registry()

    def test_dflash_registers_the_multimodal_streaming_contract(self):
        registration = self.registry.resolve("dflash")
        contract = registration.spec.feature_contract(
            FeatureMode.STREAMING, "multimodal"
        )
        self.assertEqual(
            set(contract.required_tensors),
            {"input_ids", "loss_mask", "hidden_states"},
        )
        provider = registration.providers.server_streaming_for("multimodal")
        self.assertEqual(provider.capture_method, "dflash")
        self.assertEqual(provider.layout.aux_feature, "hidden_states")

    def test_other_builtins_have_no_multimodal_contract(self):
        for name in ("domino", "eagle3", "peagle"):
            with self.subTest(algorithm=name):
                registration = self.registry.resolve(name)
                with self.assertRaises(KeyError):
                    registration.spec.feature_contract(
                        FeatureMode.STREAMING, "multimodal"
                    )

    def test_input_adapter_factory_builds_a_valid_adapter(self):
        registration = self.registry.resolve("dflash")
        provider = registration.providers.server_streaming_for("multimodal")
        adapter = provider.create_input_adapter(
            SimpleNamespace(model=SimpleNamespace(), data=SimpleNamespace())
        )
        self.assertIsInstance(adapter, VlmServerInputAdapter)


class VlmExpansionMathTest(unittest.TestCase):
    def test_image_token_count_uses_merge_size(self):
        self.assertEqual(_image_token_count([[2, 4, 6]], merge_size=2), 12)
        self.assertEqual(_image_token_count([[1, 2, 2]], merge_size=1), 4)

    def test_expand_image_region_splices_ids_and_zero_mask(self):
        ids, mask = _expand_image_region(
            [10, 99, 20],
            [0, 0, 1],
            pad_token_id=99,
            count=4,
            source="test",
        )
        self.assertEqual(ids, [10, 99, 99, 99, 99, 20])
        self.assertEqual(mask, [0, 0, 0, 0, 0, 1])

    def test_expand_image_region_requires_exactly_one_placeholder(self):
        with self.assertRaises(ValueError):
            _expand_image_region([10, 20], [1, 1], pad_token_id=99, count=4, source="t")
        with self.assertRaises(ValueError):
            _expand_image_region(
                [99, 10, 99], [0, 0, 0], pad_token_id=99, count=4, source="t"
            )


class ExtractImageFieldTest(unittest.TestCase):
    def _extract(self, record):
        from specforge.data.vlm_preprocessing import _extract_image_field

        return _extract_image_field(record, source="t")

    def test_image_and_image_path_fields(self):
        self.assertEqual(self._extract({"image": "a.jpg"}), "a.jpg")
        self.assertEqual(self._extract({"image_path": "b.jpg"}), "b.jpg")
        self.assertIsNone(self._extract({}))

    def test_images_list_takes_the_single_element(self):
        self.assertEqual(self._extract({"images": ["a.jpg"]}), "a.jpg")
        self.assertIsNone(self._extract({"images": []}))

    def test_multi_image_sample_is_fatal(self):
        from specforge.data.vlm_preprocessing import ImageDataError

        with self.assertRaises(ImageDataError):
            self._extract({"images": ["a.jpg", "b.jpg"]})

    def test_non_list_images_and_non_string_element_are_fatal(self):
        from specforge.data.vlm_preprocessing import ImageDataError

        with self.assertRaises(ImageDataError):
            self._extract({"images": "a.jpg"})
        with self.assertRaises(ImageDataError):
            self._extract({"images": [123]})

    def test_conflicting_image_fields_are_fatal(self):
        from specforge.data.vlm_preprocessing import ImageDataError

        with self.assertRaises(ImageDataError):
            self._extract({"image": "a.jpg", "images": ["b.jpg"]})

    def test_unreadable_image_is_fatal_not_skipped(self):
        from specforge.data.vlm_preprocessing import ImageDataError, _load_image

        with self.assertRaises(ImageDataError):
            _load_image("/nonexistent/path/to/image.jpg", source="t")
        with self.assertRaises(ImageDataError):
            _load_image("not-valid-base64!!!", source="t")

    def test_load_image_returns_data_uri(self):
        import base64 as b64mod

        from specforge.data.vlm_preprocessing import _load_image

        # 1x1 white PNG
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4"
            "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
        )
        _, uri = _load_image(png_b64, source="t")
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        raw = b64mod.b64decode(uri.split(",", 1)[1])
        self.assertEqual(raw, b64mod.b64decode(png_b64))

    def test_load_image_preserves_input_media_type(self):
        from specforge.data.vlm_preprocessing import _load_image

        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4"
            "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
        )
        _, uri = _load_image(f"data:image/png;base64,{png_b64}", source="t")
        self.assertTrue(uri.startswith("data:image/png;base64,"))


class VlmRequestInputsTest(unittest.TestCase):
    def test_build_request_inputs_uses_collapsed_ids_and_image_data(self):
        adapter = VlmServerInputAdapter(config=SimpleNamespace())
        tasks = [
            SimpleNamespace(
                payload={
                    "input_ids": [1, 2, 2, 2, 3],
                    "request_input_ids": [1, 2, 3],
                    "image_data": "aGVsbG8=",
                }
            ),
            SimpleNamespace(
                payload={
                    "input_ids": [7, 8],
                    "request_input_ids": [7, 8],
                    "image_data": None,
                }
            ),
        ]
        request = adapter.build_request_inputs(tasks)
        self.assertEqual(request["input_ids"], [[1, 2, 3], [7, 8]])
        self.assertEqual(request["image_data"], ["aGVsbG8=", None])


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class DraftPositionsTest(unittest.TestCase):
    """Drafts always train on the plain 1D position convention, identical to
    the text-only path; multimodal capture stores no position ids."""

    def _build_model(self):
        import torch as t
        from torch import nn

        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        class _StubDraftModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.recorded = {}
                self.sliding_window = None

            def forward(
                self,
                position_ids=None,
                noise_embedding=None,
                target_hidden=None,
                attention_mask=None,
            ):
                self.recorded["position_ids"] = position_ids
                return t.zeros(1)

        return OnlineDFlashModel(
            draft_model=_StubDraftModel(),
            target_lm_head=nn.Linear(8, 32, bias=False),
            target_embed_tokens=nn.Embedding(32, 8),
            mask_token_id=31,
            block_size=2,
            attention_backend="sdpa",
            num_anchors=4,
        )

    def test_positions_follow_the_1d_convention(self):
        import torch as t

        model = self._build_model()
        b, s = 2, 8
        input_ids = t.randint(0, 31, (b, s))
        hidden_states = t.randn(b, s, 16)
        loss_mask = t.ones(b, s)
        t.manual_seed(0)
        model._forward_draft_blocks(input_ids, hidden_states, loss_mask)
        got = model.draft_model.recorded["position_ids"]
        self.assertEqual(got.ndim, 2)
        self.assertTrue(t.equal(got[:, :s], t.arange(s).unsqueeze(0).expand(b, -1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
