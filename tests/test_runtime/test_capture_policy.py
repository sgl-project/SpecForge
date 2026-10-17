# coding=utf-8
"""Gate: the (algorithm x backend) engine matrix collapses onto policies.

The per-algorithm target-capture/load code lives once, in
``target_capture_policy``; the legacy per-algorithm leaf classes delegate to it
(their behavior gates are the existing ``test_phase_b_gate`` / backend-parity
tests), and a NEW algorithm gets hf/sglang/custom engines by registering a
target-capture policy — no engine classes.
"""

import ast
import os
import unittest

import torch

from specforge.inference.target_engine import (
    CAPTURE_POLICIES,
    TARGET_CAPTURE_POLICIES,
    CapturePolicy,
    CaptureSpec,
    DFlashCapturePolicy,
    Eagle3CapturePolicy,
    HFTargetEngine,
    TargetCaptureBatch,
    TargetCapturePolicy,
    TargetCaptureSpec,
    available_target_engines,
    get_target_engine,
    register_target_capture_policy,
    resolve_capture_policy,
    resolve_target_capture_policy,
)
from specforge.inference.target_engine.target_capture_policy import (
    DFlashTargetOutput,
    Eagle3TargetOutput,
)

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
        # DFlash now also wraps the sglang logits processor, so it can surface the
        # post-norm final hidden (last_hidden_states) for DSpark's L1 / confidence
        # losses. It still requests no vocab logits, so no eagle3 vocab mapping.
        self.assertTrue(df.sglang_build_kwargs["wrap_eagle3_logits"])
        self.assertFalse(df.sglang_strict_capture_layers)

    def test_legacy_policy_names_are_aliases(self):
        self.assertIs(CaptureSpec, TargetCaptureSpec)
        self.assertIs(CapturePolicy, TargetCapturePolicy)
        self.assertIs(CAPTURE_POLICIES, TARGET_CAPTURE_POLICIES)
        self.assertIs(
            resolve_capture_policy("eagle3"), resolve_target_capture_policy("eagle3")
        )

    def test_legacy_module_path_is_a_shim(self):
        # the old module name keeps working and resolves to the same objects
        from specforge.inference.target_engine import capture_policy as legacy

        self.assertIs(legacy.TargetCapturePolicy, TargetCapturePolicy)
        self.assertIs(legacy.CAPTURE_POLICIES, TARGET_CAPTURE_POLICIES)
        self.assertIs(legacy.Eagle3TargetOutput, Eagle3TargetOutput)

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


class LegacyDelegationTest(unittest.TestCase):
    """The legacy leaves route through the same policy code (one implementation)."""

    def test_hf_eagle3_leaf_delegates_to_policy(self):
        from specforge.inference.target_engine.eagle3_target_model import (
            HFEagle3TargetEngine,
        )

        calls = {}

        class SpyPolicy(Eagle3CapturePolicy):
            def hf_capture(self, model, layers, *a, **k):
                calls["layers"] = layers
                return Eagle3TargetOutput(
                    hidden_states=torch.zeros(1),
                    target=torch.zeros(1),
                    loss_mask=torch.zeros(1),
                    input_ids=torch.zeros(1),
                    attention_mask=torch.zeros(1),
                )

        import specforge.inference.target_engine.eagle3_target_model as m

        engine = HFEagle3TargetEngine.__new__(HFEagle3TargetEngine)
        engine.aux_hidden_states_layers = [1, 5, 9]
        engine.model = torch.nn.Linear(2, 2)
        original = m._EAGLE3
        m._EAGLE3 = SpyPolicy()
        try:
            out = engine.capture(
                input_ids=torch.zeros(1, 3, dtype=torch.long),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                loss_mask=torch.ones(1, 3, dtype=torch.long),
            )
        finally:
            m._EAGLE3 = original
        self.assertIsInstance(out, Eagle3TargetOutput)
        self.assertEqual(calls["layers"], [1, 5, 9])


if __name__ == "__main__":
    unittest.main()
