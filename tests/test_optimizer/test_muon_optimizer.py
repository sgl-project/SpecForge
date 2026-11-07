"""BF16Optimizer ``optimizer='muon'`` (single process, CPU).

* Newton-Schulz is ``torch.optim.Muon``'s op for op; the whole update equals
  ``torch.optim.Muon`` + ``torch.optim.AdamW`` under the global clip, bit for
  bit, for every Muon setting.
* Group membership: every 2-D nn.Linear weight in a decoder layer
  (``layers.<i>.`` / ``midlayer.``) is Muon, everything else AdamW.
* One LR multiplier, two peak rates; ``lr`` stays the AdamW rate.
* Save / resume continues bit for bit; optimizer-kind, membership and
  hyperparameter changes are refused.
* The default AdamW path equals ``torch.optim.AdamW`` bit for bit and keeps
  its checkpoint layout.
"""

from __future__ import annotations

import copy
import io
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _muon_helpers import (  # noqa: E402
    HAS_TORCH_MUON,
    TorchReference,
    constant_warmup_lr,
    expected_muon_names,
    tiny_draft,
    tiny_inputs,
    tree_differences,
)

from specforge.muon import (  # noqa: E402
    NS_COEFFICIENTS,
    ShardedMuon,
    assign_owners,
    capture_parameter_layout,
    muon_lr_ratio,
    newton_schulz_cost,
    zeropower_via_newtonschulz5,
)
from specforge.optimizer import BF16Optimizer  # noqa: E402

LR, MUON_LR = 1e-3, 3e-2
# Warmup 2 of 8 updates, then constant.
TOTAL, W = 8, 2
CONSTANT = dict(lr_scheduler="constant", warmup_ratio=W / TOTAL, total_steps=TOTAL)
ADAMW_STATE_KEYS = {
    "optimizer_state_dict",
    "scheduler_state_dict",
    "lr_scheduler_type",
    "max_grad_norm",
    "fp32_params",
}


def _muon(model, **overrides):
    kwargs = dict(
        lr=LR, max_grad_norm=1.0, optimizer="muon", muon_lr=MUON_LR, **CONSTANT
    )
    kwargs.update(overrides)
    return BF16Optimizer(model, **kwargs)


def _quiet():
    return mock.patch("specforge.optimizer.print_on_rank0")


class _ProcessGroup:
    """BF16Optimizer.load_state_dict logs through print_on_rank0 (needs a group)."""

    created = False

    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist

        if dist.is_available() and not dist.is_initialized():
            store = dist.FileStore(
                os.path.join(tempfile.mkdtemp(prefix="muon_pg_"), "store"), 1
            )
            dist.init_process_group("gloo", store=store, rank=0, world_size=1)
            cls.created = True

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist

        if cls.created:
            dist.destroy_process_group()
            cls.created = False


class TestNewtonSchulz(unittest.TestCase):
    @unittest.skipUnless(
        HAS_TORCH_MUON, "torch.optim.Muon (torch >= 2.9) is the oracle"
    )
    def test_matches_torch_muon_op_for_op(self):
        from torch.optim._muon import _zeropower_via_newtonschulz

        generator = torch.Generator().manual_seed(0)
        for shape in ((7, 5), (5, 7), (6, 6), (33, 17), (17, 33), (1, 9), (9, 1)):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(shape=shape, dtype=dtype):
                    grad = torch.randn(shape, generator=generator).to(dtype)
                    expected = _zeropower_via_newtonschulz(
                        grad.clone(), NS_COEFFICIENTS, 5, 1e-7
                    )
                    actual = zeropower_via_newtonschulz5(grad.clone(), ns_steps=5)
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertEqual(actual.shape, grad.shape)
                    self.assertTrue(torch.equal(actual, expected))

    def test_orthogonalizes(self):
        update = torch.randn(64, 16, generator=torch.Generator().manual_seed(1))
        ortho = zeropower_via_newtonschulz5(update, ns_steps=5).float()
        singular = torch.linalg.svdvals(ortho)
        # Quintic NS: singular values land in roughly (0.5, 1.5), not exactly 1.
        self.assertGreater(float(singular.min()), 0.4)
        self.assertLess(float(singular.max()), 1.3)

    def test_rejects_bad_input(self):
        with self.assertRaisesRegex(ValueError, "2-D"):
            zeropower_via_newtonschulz5(torch.randn(4), ns_steps=5)
        with self.assertRaisesRegex(ValueError, "ns_steps"):
            zeropower_via_newtonschulz5(torch.randn(4, 4), ns_steps=0)

    def test_lr_ratios_match_torch_muon(self):
        self.assertEqual(muon_lr_ratio((12, 8)), (12 / 8) ** 0.5)
        self.assertEqual(muon_lr_ratio((8, 12)), 1.0)
        self.assertEqual(muon_lr_ratio((8, 12), "match_rms_adamw"), 0.2 * 12**0.5)
        with self.assertRaisesRegex(ValueError, "adjust_lr_fn"):
            muon_lr_ratio((8, 12), "spectral")


class TestMembership(unittest.TestCase):
    def test_layer_matrices_are_muon_and_everything_else_adamw(self):
        model = tiny_draft()
        layout = capture_parameter_layout(model)
        muon = [name for name, info in layout.items() if info.muon]
        self.assertEqual(muon, expected_muon_names())
        adamw = [name for name, info in layout.items() if not info.muon]
        for name in (
            "fc.weight",
            "hidden_norm.weight",
            "norm.weight",
            "embed_proj.0.weight",
            "embed_proj.2.weight",
            "prefix_gru.weight_ih_l0",
            "prefix_gru.weight_hh_l0",
            "layers.0.self_attn.q_norm.weight",
            "layers.1.input_layernorm.weight",
        ):
            self.assertIn(name, adamw)
        self.assertNotIn("frozen_table", layout)  # frozen: in neither group
        self.assertEqual(layout["layers.0.self_attn.o_proj.weight"].shape, (8, 12))
        optimizer = _muon(model)
        membership = optimizer.muon_membership()
        self.assertEqual(
            [name for name, _ in membership["muon"]], expected_muon_names()
        )
        self.assertEqual(
            len(membership["muon"]) + len(membership["adamw"]),
            len(optimizer.fp32_params),
        )
        self.assertEqual(
            membership["muon_numel"] + membership["adamw_numel"],
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    def test_decoder_layer_prefixes_of_every_draft_family(self):
        # DFlash family / P-EAGLE: layers.<i>; MTP: mtp.layers.<i>; EAGLE3: midlayer.
        for path in ("layers", "mtp.layers", "midlayer"):
            with self.subTest(path=path):
                model = nn.Module()
                model.fc = nn.Linear(4, 4, bias=False)
                if path == "midlayer":
                    model.midlayer = nn.Sequential(nn.Linear(4, 4, bias=False))
                    expected = {"midlayer.0.weight"}
                elif path == "mtp.layers":
                    model.mtp = nn.Module()
                    model.mtp.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
                    expected = {"mtp.layers.0.weight"}
                else:
                    model.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
                    expected = {"layers.0.weight"}
                layout = capture_parameter_layout(model)
                self.assertEqual(
                    {name for name, info in layout.items() if info.muon}, expected
                )

    def test_biases_and_linear_weights_outside_layers_stay_on_adamw(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(4, 3, bias=True)])
        model.head = nn.Linear(3, 5)
        model.sublayers = nn.ModuleList([nn.Linear(3, 3)])
        layout = capture_parameter_layout(model)
        self.assertEqual(
            {name for name, info in layout.items() if info.muon}, {"layers.0.weight"}
        )

    def test_a_weight_shared_with_a_non_layer_module_stays_on_adamw(self):
        for head in (nn.Linear(4, 4, bias=False), nn.Embedding(4, 4)):
            with self.subTest(head=type(head).__name__):
                model = nn.Module()
                model.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
                model.lm_head = head
                model.lm_head.weight = model.layers[0].weight
                layout = capture_parameter_layout(model)
                self.assertEqual(list(layout), ["layers.0.weight"])
                self.assertFalse(layout["layers.0.weight"].muon)

    def test_wrapper_segments_are_canonicalized(self):
        layout = capture_parameter_layout(tiny_draft())

        class Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self._fsdp_wrapped_module = module

        wrapped = tiny_draft()
        wrapped.layers[0] = Wrapper(wrapped.layers[0])
        self.assertEqual(capture_parameter_layout(wrapped), layout)

    def test_flattened_parameters_need_the_pre_wrap_layout(self):
        model = tiny_draft()
        layout = capture_parameter_layout(model)
        flat = model.layers[0].self_attn.q_proj.weight
        flat.data = flat.data.reshape(-1)
        with self.assertRaisesRegex(ValueError, "before FSDP flattens"):
            _muon(model)
        # with the captured layout it builds (the backend path)
        optimizer = _muon(model, parameter_layout=layout)
        self.assertEqual(optimizer.muon_optimizer.shapes[0], (12, 8))

    def test_startup_log_lists_both_groups(self):
        with self.assertLogs("specforge.optimizer", level=logging.INFO) as logs:
            _muon(tiny_draft())
        text = "\n".join(logs.output)
        self.assertIn("14 matrices", text)
        self.assertIn("muon: layers.0.self_attn.q_proj.weight [12, 8]", text)
        self.assertIn("adamw: fc.weight [8, 24]", text)
        self.assertIn("adjust_lr_fn=original", text)

    def test_owner_assignment_is_deterministic_and_balanced(self):
        # 5 layers x (q, k, v, o, gate, up, down) on 16 ranks.
        per_layer = [
            (4096, 5120),
            (1024, 5120),
            (1024, 5120),
            (5120, 4096),
            (17408, 5120),
            (17408, 5120),
            (5120, 17408),
        ]
        shapes = per_layer * 5
        owners = assign_owners(shapes, 16)
        self.assertEqual(owners, assign_owners(shapes, 16))
        loads = [0] * 16
        for shape, owner in zip(shapes, owners):
            loads[owner] += newton_schulz_cost(shape)
        biggest = max(newton_schulz_cost(shape) for shape in shapes)
        self.assertLessEqual(max(loads), biggest + newton_schulz_cost((4096, 5120)))
        self.assertEqual(len(set(owners)), 16)


@unittest.skipUnless(HAS_TORCH_MUON, "torch.optim.Muon (torch >= 2.9) is the oracle")
class TestMatchesTorchOptim(unittest.TestCase):
    def _run(
        self,
        *,
        steps=TOTAL,
        max_grad_norm=1.0,
        nesterov=True,
        momentum=0.95,
        muon_weight_decay=0.0,
        adjust_lr_fn="original",
    ):
        model = tiny_draft()
        reference_model = copy.deepcopy(model)
        optimizer = _muon(
            model,
            max_grad_norm=max_grad_norm,
            muon_nesterov=nesterov,
            muon_momentum=momentum,
            muon_weight_decay=muon_weight_decay,
            muon_adjust_lr_fn=adjust_lr_fn,
        )
        reference = TorchReference(
            reference_model,
            lr=LR,
            muon_lr=MUON_LR,
            momentum=momentum,
            nesterov=nesterov,
            muon_weight_decay=muon_weight_decay,
            adjust_lr_fn=adjust_lr_fn,
            max_grad_norm=max_grad_norm,
            warmup=W,
        )
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        for update in range(1, steps + 1):
            x = tiny_inputs(update)
            model(x).backward()
            reference_model(x).backward()
            rates = optimizer.learning_rates_by_group()
            self.assertEqual(rates["adamw"], constant_warmup_lr(update, LR, W))
            self.assertEqual(rates["muon"], constant_warmup_lr(update, MUON_LR, W))
            self.assertEqual(optimizer.get_learning_rate(), rates["adamw"])
            norm = optimizer.step()
            reference_norm = reference.step(update)
            self.assertTrue(torch.equal(norm, reference_norm))
            for name, master in zip(names, optimizer.fp32_params):
                self.assertTrue(
                    torch.equal(master, reference.master(name)), (update, name)
                )
            reference_weights = dict(reference_model.named_parameters())
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p, reference_weights[name]))
            for index, name in enumerate(expected_muon_names()):
                master = optimizer.muon_optimizer.param_groups[0]["params"][index]
                self.assertTrue(
                    torch.equal(
                        optimizer.muon_optimizer.state[master]["momentum_buffer"],
                        reference.momentum(name),
                    )
                )
        return optimizer

    def test_default_muon_matches_torch_muon_and_adamw_bitwise(self):
        optimizer = self._run()
        self.assertGreater(float(optimizer.last_grad_norm), 0.0)

    def test_every_muon_setting_matches(self):
        for kwargs in (
            dict(nesterov=False),
            dict(momentum=0.9),
            dict(muon_weight_decay=0.1),
            dict(max_grad_norm=1e9),
            dict(adjust_lr_fn="match_rms_adamw"),
        ):
            with self.subTest(**kwargs):
                self._run(steps=4, **kwargs)

    def test_lr_schedule_is_one_multiplier_with_two_peaks(self):
        optimizer = _muon(tiny_draft())
        self.assertEqual(optimizer.scheduler.base_lrs, [LR, MUON_LR])
        for update in range(1, TOTAL + 1):
            self.assertEqual(
                optimizer.learning_rates_by_group(),
                {
                    "adamw": constant_warmup_lr(update, LR, W),
                    "muon": constant_warmup_lr(update, MUON_LR, W),
                },
            )
            optimizer.scheduler.step()
        cosine = _muon(tiny_draft(), lr_scheduler="cosine")
        for _ in range(TOTAL - 1):
            cosine.scheduler.step()
            rates = cosine.learning_rates_by_group()
            self.assertAlmostEqual(rates["muon"] / rates["adamw"], MUON_LR / LR)

    def test_shard_plan_is_local_without_a_process_group(self):
        optimizer = _muon(tiny_draft())
        optimizer.configure_grad_norm_reduction(enabled=False)
        self.assertEqual(optimizer.muon_optimizer.plan.describe()["mode"], "local")


class TestResume(_ProcessGroup, unittest.TestCase):
    STEPS, CUT = TOTAL, 3

    def _train(self, model, optimizer, first, last):
        for update in range(first, last + 1):
            model(tiny_inputs(update)).backward()
            optimizer.step()

    @staticmethod
    def _roundtrip(state):
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        return torch.load(buffer, weights_only=False)

    def test_resume_continues_bitwise(self):
        straight = tiny_draft()
        straight_optimizer = _muon(straight)
        self._train(straight, straight_optimizer, 1, self.STEPS)

        first = tiny_draft()
        first_optimizer = _muon(first)
        self._train(first, first_optimizer, 1, self.CUT)
        saved_weights = {k: v.clone() for k, v in first.state_dict().items()}
        saved = self._roundtrip(first_optimizer.state_dict())
        self.assertEqual(saved["optimizer_kind"], "muon")
        self.assertEqual(set(saved["optimizer_state_dict"]), {"adamw", "muon"})

        # different init: everything comes from the checkpoint
        resumed = tiny_draft(seed=7)
        resumed.load_state_dict(saved_weights)
        resumed_optimizer = _muon(resumed)
        with _quiet():
            resumed_optimizer.load_state_dict(saved)
        self.assertEqual(
            resumed_optimizer.learning_rates_by_group(),
            {
                "adamw": constant_warmup_lr(self.CUT + 1, LR, W),
                "muon": constant_warmup_lr(self.CUT + 1, MUON_LR, W),
            },
        )
        self._train(resumed, resumed_optimizer, self.CUT + 1, self.STEPS)
        self.assertEqual(
            tree_differences(
                resumed_optimizer.state_dict(), straight_optimizer.state_dict()
            ),
            [],
        )
        self.assertEqual(
            tree_differences(resumed.state_dict(), straight.state_dict()), []
        )

    def test_resume_refuses_an_optimizer_change(self):
        muon_state = _muon(tiny_draft()).state_dict()
        adamw_state = BF16Optimizer(
            tiny_draft(), lr=LR, max_grad_norm=1.0, **CONSTANT
        ).state_dict()
        with self.assertRaisesRegex(
            ValueError, "optimizer is 'muon' but this run has optimizer='adamw'"
        ):
            BF16Optimizer(
                tiny_draft(), lr=LR, max_grad_norm=1.0, **CONSTANT
            ).load_state_dict(muon_state)
        with self.assertRaisesRegex(
            ValueError, "optimizer is 'adamw' but this run has optimizer='muon'"
        ):
            _muon(tiny_draft()).load_state_dict(adamw_state)
        retired = dict(muon_state, optimizer_type="muon")
        with self.assertRaisesRegex(ValueError, "retired draft hybrid-Muon format"):
            _muon(tiny_draft()).load_state_dict(retired)

    def test_resume_refuses_changed_hyperparameters_and_membership(self):
        state = _muon(tiny_draft()).state_dict()
        for change in (
            dict(muon_lr=6e-2),
            dict(muon_momentum=0.9),
            dict(muon_nesterov=False),
            dict(muon_ns_steps=4),
            dict(muon_weight_decay=0.1),
            dict(muon_adjust_lr_fn="match_rms_adamw"),
            dict(lr=7e-4),
        ):
            with (
                self.subTest(**change),
                self.assertRaisesRegex(ValueError, "hyperparameters"),
            ):
                _muon(tiny_draft(), **change).load_state_dict(state)
        other = tiny_draft()
        other.layers[1].requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "membership"):
            _muon(other).load_state_dict(state)
        broken = copy.deepcopy(state)
        del broken["muon"]
        with self.assertRaisesRegex(ValueError, "no Muon record"):
            _muon(tiny_draft()).load_state_dict(broken)

    def test_nonfinite_gradient_mutates_nothing(self):
        model = tiny_draft()
        optimizer = _muon(model)
        self._train(model, optimizer, 1, 2)
        before = copy.deepcopy(optimizer.state_dict())
        weights = copy.deepcopy(model.state_dict())
        model(tiny_inputs(3)).backward()
        model.layers[0].mlp.up_proj.weight.grad[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        self.assertEqual(tree_differences(optimizer.state_dict(), before), [])
        self.assertEqual(tree_differences(model.state_dict(), weights), [])


class TestConstruction(unittest.TestCase):
    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "offload_master"):
            _muon(tiny_draft(), offload_master=True)
        with self.assertRaisesRegex(ValueError, "positive muon_lr"):
            _muon(tiny_draft(), muon_lr=None)
        with self.assertRaisesRegex(ValueError, "unsupported optimizer"):
            BF16Optimizer(tiny_draft(), lr=LR, optimizer="sgd")
        with self.assertRaisesRegex(ValueError, "adjust_lr_fn"):
            _muon(tiny_draft(), muon_adjust_lr_fn="spectral")
        no_layers = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        with self.assertRaisesRegex(ValueError, "found no"):
            _muon(no_layers)
        only_layers = nn.Module()
        only_layers.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        with self.assertRaisesRegex(ValueError, "AdamW group to be non-empty"):
            _muon(only_layers)

    def test_sharded_slices_need_a_process_group(self):
        model = tiny_draft()
        layout = capture_parameter_layout(model)
        weight = model.layers[0].mlp.gate_proj.weight
        weight.data = weight.data.reshape(-1)[:50].clone()  # a local slice
        optimizer = _muon(model, parameter_layout=layout)
        self.assertIsNone(optimizer.muon_optimizer.plan)
        with self.assertRaisesRegex(RuntimeError, "no sharded process group"):
            optimizer.configure_grad_norm_reduction(enabled=False)

    def test_muon_rejects_bad_hyperparameters(self):
        p = [torch.zeros(4, 4)]
        for kwargs, match in (
            (dict(lr=0.0), "learning rate"),
            (dict(lr=1.0, momentum=1.0), "momentum"),
            (dict(lr=1.0, ns_steps=0), "ns_steps"),
            (dict(lr=1.0, weight_decay=-1.0), "weight decay"),
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, match):
                ShardedMuon(p, names=["w"], shapes=[(4, 4)], **kwargs)


class TestAdamWPathUnchanged(_ProcessGroup, unittest.TestCase):
    """The default path is plain ``torch.optim.AdamW`` and keeps its layout."""

    def test_default_path_matches_torch_adamw_bitwise(self):
        model = tiny_draft()
        reference_model = copy.deepcopy(model)
        optimizer = BF16Optimizer(model, lr=LR, max_grad_norm=1.0, **CONSTANT)
        reference = TorchReference(
            reference_model,
            lr=LR,
            muon_lr=MUON_LR,
            max_grad_norm=1.0,
            warmup=W,
            use_muon=False,
        )
        self.assertEqual(optimizer.optimizer_kind, "adamw")
        self.assertIs(type(optimizer.optimizer), torch.optim.AdamW)
        self.assertEqual(optimizer.learning_rates_by_group(), {})
        for update in range(1, 6):
            x = tiny_inputs(update)
            model(x).backward()
            reference_model(x).backward()
            self.assertTrue(torch.equal(optimizer.step(), reference.step(update)))
            reference_weights = dict(reference_model.named_parameters())
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p, reference_weights[name]), name)
        self.assertEqual(set(optimizer.state_dict()), ADAMW_STATE_KEYS)

    def test_adamw_checkpoint_round_trips(self):
        model = tiny_draft()
        optimizer = BF16Optimizer(model, lr=LR, max_grad_norm=1.0, **CONSTANT)
        for update in (1, 2):
            model(tiny_inputs(update)).backward()
            optimizer.step()
        restored = BF16Optimizer(
            copy.deepcopy(model), lr=LR, max_grad_norm=1.0, **CONSTANT
        )
        with _quiet():
            restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertEqual(
            tree_differences(restored.state_dict(), optimizer.state_dict()), []
        )


class TestFactoryAndSchema(unittest.TestCase):
    MINIMAL = {
        "model": {"target_model_path": "target", "draft_model_config": "draft.json"},
        "data": {"hidden_states_path": "/features"},
    }

    def _config(self, **training):
        from specforge.config import Config

        payload = copy.deepcopy(self.MINIMAL)
        payload["training"] = {"max_steps": TOTAL, "learning_rate": LR, **training}
        return Config.model_validate(payload)

    def test_schema(self):
        from pydantic import ValidationError

        from specforge.config import Config

        default = self._config().training
        self.assertEqual(
            (
                default.optimizer,
                default.muon_learning_rate,
                default.muon_momentum,
                default.muon_nesterov,
                default.muon_ns_steps,
                default.muon_weight_decay,
                default.muon_adjust_lr_fn,
            ),
            ("adamw", None, 0.95, True, 5, 0.0, "original"),
        )
        muon = self._config(optimizer="muon", muon_learning_rate=MUON_LR).training
        self.assertEqual((muon.optimizer, muon.muon_learning_rate), ("muon", MUON_LR))
        # dump -> validate round trips (overrides, role split) keep both valid
        for cfg in (
            self._config(),
            self._config(optimizer="muon", muon_learning_rate=MUON_LR),
        ):
            Config.model_validate(cfg.model_dump())
        for training, match in (
            (dict(optimizer="muon"), "requires training.muon_learning_rate"),
            (
                dict(
                    optimizer="muon",
                    muon_learning_rate=MUON_LR,
                    optimizer_cpu_offload=True,
                ),
                "optimizer_cpu_offload",
            ),
            (dict(muon_learning_rate=MUON_LR), "apply only to training.optimizer=muon"),
            (dict(muon_momentum=0.9), "apply only to training.optimizer=muon"),
            (
                dict(muon_adjust_lr_fn="match_rms_adamw"),
                "apply only to training.optimizer=muon",
            ),
            (dict(optimizer="lion"), "optimizer"),
            (dict(optimizer="muon", muon_learning_rate=0.0), "muon_learning_rate"),
            (
                dict(optimizer="muon", muon_learning_rate=MUON_LR, muon_momentum=1.0),
                "muon_momentum",
            ),
            (
                dict(optimizer="muon", muon_learning_rate=MUON_LR, muon_ns_steps=0),
                "muon_ns_steps",
            ),
        ):
            with (
                self.subTest(**training),
                self.assertRaisesRegex(ValidationError, match),
            ):
                self._config(**training)

    def test_factory_builds_each_optimizer(self):
        from specforge.training.assembly import _ConfiguredOptimizerFactory

        adamw = _ConfiguredOptimizerFactory(self._config(lr_scheduler="constant"))
        adamw.capture_parameter_layout(tiny_draft())
        self.assertIsNone(adamw.parameter_layout)
        self.assertEqual(adamw(tiny_draft()).optimizer_kind, "adamw")

        cfg = self._config(
            lr_scheduler="constant",
            optimizer="muon",
            muon_learning_rate=MUON_LR,
            muon_momentum=0.9,
            muon_nesterov=False,
            muon_ns_steps=4,
            muon_weight_decay=0.01,
            muon_adjust_lr_fn="match_rms_adamw",
        )
        factory = _ConfiguredOptimizerFactory(cfg)
        model = tiny_draft()
        factory.capture_parameter_layout(model)
        self.assertEqual(factory.parameter_layout, capture_parameter_layout(model))
        optimizer = factory(model)
        group = optimizer.muon_optimizer.param_groups[0]
        self.assertEqual(
            (
                group["initial_lr"],
                group["momentum"],
                group["nesterov"],
                group["ns_steps"],
                group["weight_decay"],
                group["adjust_lr_fn"],
            ),
            (MUON_LR, 0.9, False, 4, 0.01, "match_rms_adamw"),
        )
        self.assertEqual(optimizer.scheduler.base_lrs, [LR, MUON_LR])


class TestRealDrafts(unittest.TestCase):
    """The rule on the shipped DFlash-family drafts."""

    def test_dspark_backbone_on_muon_heads_on_adamw(self):
        from transformers import Qwen3Config

        from specforge.modeling.draft.dspark import DSparkDraftModel

        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=128,
            block_size=4,
            num_target_layers=4,
            dflash_config={
                "attention_mode": "gqa",
                "projector_type": "dspark",
                "target_layer_ids": [0],
                "markov_rank": 4,
                "markov_head_type": "vanilla",
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
            },
        )
        layout = capture_parameter_layout(DSparkDraftModel(config))
        muon = {name for name, info in layout.items() if info.muon}
        self.assertEqual(
            muon,
            {
                f"layers.{i}.{name}.weight"
                for i in range(2)
                for name in (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                )
            },
        )
        for name in (
            "fc.weight",
            "markov_head.markov_w2.weight",
            "confidence_head.proj.weight",
        ):
            self.assertFalse(layout[name].muon, name)

    def test_dflash2_layer_convolutions_on_muon_selector_on_adamw(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_modeling"))
        from test_dflash2 import _tiny_config

        from specforge.modeling.draft.dflash2 import DFlash2DraftModel

        layout = capture_parameter_layout(DFlash2DraftModel(_tiny_config()))
        muon = {name for name, info in layout.items() if info.muon}
        self.assertIn("layers.0.self_attn.q_proj.weight", muon)
        self.assertIn("layers.0.mlp.down_proj.weight", muon)
        # Zero-initialized per-layer convolution projections are layer matrices.
        self.assertIn("layers.0.attention_conv.kernel_projection.weight", muon)
        self.assertIn("layers.0.mlp_conv.kernel_projection.weight", muon)
        for name in (
            "fc.weight",
            "layers.0.attention_conv.base_kernel",
            "candidate_selector.hidden_projection.weight",
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
        ):
            self.assertFalse(layout[name].muon, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
