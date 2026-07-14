# coding=utf-8
"""Gate: the (algorithm x backend) engine matrix collapses onto policies.

Algorithm-specific target capture lives in ``target_capture_policy``. A new
algorithm gets the generic HF/SGLang/custom engines by registering one policy;
it does not add backend-specific engine classes.
"""

import ast
import os
import unittest
from types import SimpleNamespace

import torch

from specforge.inference.target_engine import (
    TARGET_CAPTURE_POLICIES,
    DFlashCapturePolicy,
    Eagle3CapturePolicy,
    HFTargetEngine,
    TargetCaptureBatch,
    TargetCapturePolicy,
    TargetCaptureSpec,
    available_target_engines,
    get_target_engine,
    register_target_capture_policy,
    resolve_target_capture_policy,
)
from specforge.inference.target_engine.target_capture_policy import (
    DFlashTargetOutput,
    Eagle3TargetOutput,
)
from specforge.inference.media import MediaInputs

_ENGINE_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "specforge", "inference", "target_engine"
    )
)


class PolicyRegistryTest(unittest.TestCase):
    def test_builtin_policies_registered(self):
        self.assertIsInstance(
            resolve_target_capture_policy("eagle3"), Eagle3CapturePolicy
        )
        self.assertIsInstance(
            resolve_target_capture_policy("dflash"), DFlashCapturePolicy
        )
        # Domino trains on DFlash-captured features: same policy object.
        self.assertIs(
            resolve_target_capture_policy("domino"),
            resolve_target_capture_policy("dflash"),
        )

    def test_unknown_policy_raises(self):
        with self.assertRaises(KeyError):
            resolve_target_capture_policy("does-not-exist")

    def test_specs_carry_the_backend_flags(self):
        e3 = resolve_target_capture_policy("eagle3").spec
        df = resolve_target_capture_policy("dflash").spec
        self.assertEqual(e3.num_capture_layers, 3)
        self.assertTrue(e3.sglang_build_kwargs["wrap_eagle3_logits"])
        self.assertTrue(e3.sglang_strict_capture_layers)
        self.assertFalse(df.sglang_build_kwargs["wrap_eagle3_logits"])
        self.assertFalse(df.sglang_strict_capture_layers)

    def test_policy_outputs_are_typed_capture_batches(self):
        # the policy layer's contract with the runtime adapter: typed batches
        self.assertTrue(issubclass(Eagle3TargetOutput, TargetCaptureBatch))
        self.assertTrue(issubclass(DFlashTargetOutput, TargetCaptureBatch))

    def test_eagle3_default_layers_rule(self):
        class Cfg:
            num_hidden_layers = 32

        layers = Eagle3CapturePolicy().resolve_capture_layers(Cfg(), None)
        self.assertEqual(layers, [1, 15, 28])
        with self.assertRaises(AssertionError):
            Eagle3CapturePolicy().resolve_capture_layers(None, [1, 2])

    def test_offline_sglang_capture_requests_last_hidden_states_without_logits(self):
        class RecordingBackend:
            def __init__(self):
                self.kwargs = None

            def extend_eagle3(self, input_ids, attention_mask, loss_mask, **kwargs):
                self.kwargs = kwargs
                return (
                    [[input_ids, attention_mask, loss_mask]],
                    [None],
                    [torch.ones(3, 2)],
                    [torch.ones(3, 4)],
                )

        backend = RecordingBackend()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)

        captured = Eagle3CapturePolicy().sglang_capture(
            backend,
            input_ids,
            attention_mask,
            loss_mask,
            return_last_hidden_states=True,
            return_logits=False,
        )

        self.assertEqual(
            backend.kwargs,
            {
                "return_last_hidden_states": True,
                "return_logits": False,
                "shard_returns": False,
            },
        )
        self.assertIsNone(captured.target)
        self.assertEqual(captured.hidden_states.shape, (1, 3, 2))
        self.assertEqual(captured.last_hidden_states.shape, (1, 3, 4))

    def test_vlm_sglang_capture_uses_media_extend_without_sharding(self):
        class RecordingBackend:
            def __init__(self):
                self.kwargs = None

            def extend_eagle3_vlm(
                self, input_ids, attention_mask, loss_mask, **kwargs
            ):
                self.kwargs = kwargs
                return (
                    [[input_ids, attention_mask, loss_mask]],
                    [torch.ones(3, 8)],
                    [torch.ones(3, 12)],
                    [None],
                )

        backend = RecordingBackend()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)
        grid = torch.tensor([[1, 2, 2]])
        media = MediaInputs(
            pixel_values=torch.ones(4, 8), image_grid_thw=(grid,)
        )

        captured = Eagle3CapturePolicy().sglang_capture(
            backend,
            input_ids,
            attention_mask,
            loss_mask,
            media_inputs=media,
        )

        self.assertIs(backend.kwargs["pixel_values"], media.pixel_values)
        self.assertEqual(backend.kwargs["image_grid_thw"], [grid])
        self.assertEqual(captured.hidden_states.shape, (1, 3, 12))
        with self.assertRaisesRegex(ValueError, "shard_returns"):
            Eagle3CapturePolicy().sglang_capture(
                backend,
                input_ids,
                attention_mask,
                loss_mask,
                media_inputs=media,
                shard_returns=True,
            )

    def test_vlm_hf_capture_passes_media_without_moe_only_kwargs(self):
        class RecordingVLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                language_model = torch.nn.Module()
                language_model.layers = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(3)]
                )
                self.model = torch.nn.Module()
                self.model.language_model = language_model
                self.kwargs = None

            def forward(self, **kwargs):
                self.kwargs = kwargs
                hidden = torch.ones(
                    kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1], 4
                )
                for layer in self.model.language_model.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(logits=torch.ones(*hidden.shape[:2], 8))

        model = RecordingVLM()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)
        grid = torch.tensor([[1, 2, 2]])
        media = MediaInputs(
            pixel_values=torch.ones(4, 8), image_grid_thw=(grid,)
        )

        captured = Eagle3CapturePolicy().hf_capture(
            model,
            [0, 1, 2],
            input_ids,
            attention_mask,
            loss_mask,
            media_inputs=media,
        )

        self.assertNotIn("output_router_logits", model.kwargs)
        self.assertIs(model.kwargs["pixel_values"], media.pixel_values)
        self.assertTrue(torch.equal(model.kwargs["image_grid_thw"], grid))
        self.assertEqual(captured.hidden_states.shape, (1, 3, 12))


class _RecordingPolicy(TargetCapturePolicy):
    """A minimal fake algorithm: records calls, returns sentinels."""

    spec = TargetCaptureSpec(name="fake_algo")

    def __init__(self):
        self.calls = []

    def hf_load(self, path, torch_dtype, device, cache_dir, **kwargs):
        self.calls.append(("hf_load", path))
        return torch.nn.Linear(2, 2)

    def hf_capture(
        self, model, capture_layers, input_ids, attention_mask, loss_mask, **kw
    ):
        self.calls.append(("hf_capture", capture_layers))
        return "hf_sentinel"

    def sglang_capture(self, backend, input_ids, attention_mask, loss_mask, **kw):
        self.calls.append(("sglang_capture",))
        return "sglang_sentinel"


class GenericEngineTest(unittest.TestCase):
    def _register(self, name="fake_algo"):
        policy = _RecordingPolicy()
        register_target_capture_policy(name, policy)
        self.addCleanup(TARGET_CAPTURE_POLICIES.pop, name, None)
        return policy

    def test_new_algorithm_needs_no_engine_class(self):
        policy = self._register()
        self.assertIn("fake_algo", available_target_engines())
        engine = get_target_engine("some/path", strategy="fake_algo", backend="hf")
        self.assertIsInstance(engine, HFTargetEngine)
        self.assertEqual(engine.backend, "hf")

        out = engine.capture(
            input_ids=torch.zeros(1, 3, dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            loss_mask=torch.ones(1, 3, dtype=torch.long),
        )
        self.assertEqual(out, "hf_sentinel")
        self.assertEqual([c[0] for c in policy.calls], ["hf_load", "hf_capture"])

    def test_generic_engine_threads_capture_layers(self):
        policy = self._register()
        engine = HFTargetEngine(torch.nn.Linear(2, 2), policy)
        engine.set_capture_layers([0, 1])
        engine.capture(
            input_ids=torch.zeros(1, 3, dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            loss_mask=torch.ones(1, 3, dtype=torch.long),
        )
        self.assertIn(("hf_capture", [0, 1]), policy.calls)

    def test_custom_capture_default_is_unimplemented(self):
        policy = self._register()
        with self.assertRaises(NotImplementedError):
            policy.custom_capture(None, None, None, None, None)

    def test_factory_still_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            get_target_engine("some/path", strategy="never-registered")


class PolicyModuleIsSglangFreeTest(unittest.TestCase):
    """The policy module keeps the B2 invariant: no module-level sglang imports."""

    def test_no_toplevel_sglang_import(self):
        path = os.path.join(_ENGINE_DIR, "target_capture_policy.py")
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        hits = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                hits += [n.name for n in node.names if n.name.split(".")[0] == "sglang"]
            elif isinstance(node, ast.ImportFrom):
                if (
                    node.level == 0
                    and node.module
                    and node.module.split(".")[0] == "sglang"
                ):
                    hits.append(node.module)
        self.assertEqual(
            hits, [], f"target_capture_policy leaks sglang imports: {hits}"
        )


if __name__ == "__main__":
    unittest.main()
