"""``constant_linear_decay``: scheduler, closed forms, optimizer branch load, config.

LR of 1-based update u (the value of ``param_groups[*]["lr"]`` while update u runs):
    u <= W            u / W * base                      (constant warmup)
    W < u <= D        base
    D < u <= T        base * ((T - u + 1) / (T - D))    (D+1 -> base, T -> base/(T-D))
    u > T             0.0
"""

from __future__ import annotations

import copy
import io
import os
import tempfile
import unittest
import warnings

import torch
from torch import nn

from specforge.lr_scheduler import (
    ConstantLinearDecayWarmupLR,
    ConstantWarmupLR,
    CosineAnnealingWarmupLR,
    completed_updates_from_state,
    constant_linear_decay_lr,
    constant_warmup_lr,
    warmup_updates,
)
from specforge.optimizer import BF16Optimizer

BASE = 1e-3
# Full-size numbers: 83,600-update horizon, warmup 2,000, decay 12,000 -> 16,720.
TOTAL, RATIO, D, T = 83600, 0.023923444976076555, 12000, 16720
W = 2000
# Scaled-down world: total 40, warmup 4, decay after update 10 until update 17.
S_TOTAL, S_RATIO, S_D, S_T = 40, 0.1, 10, 17
S_W = 4
STATE_KEYS = {
    "optimizer_state_dict",
    "scheduler_state_dict",
    "lr_scheduler_type",
    "max_grad_norm",
    "fp32_params",
}


def _scheduler_lrs(scheduler, optimizer, updates):
    """LR seen by updates 1..updates (read before the post-update scheduler step)."""
    lrs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scheduler.step() without optimizer.step()
        for _ in range(updates):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
    return lrs


def _adamw(lr=BASE):
    return torch.optim.AdamW([nn.Parameter(torch.zeros(2))], lr=lr)


class _Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))


def _optimizer(kind, model=None, d=S_D, t=S_T, ratio=S_RATIO):
    model = _Draft() if model is None else model
    decay = (
        dict(lr_decay_start_update=d, lr_decay_end_update=t)
        if kind == "constant_linear_decay"
        else {}
    )
    optimizer = BF16Optimizer(
        model,
        lr=BASE,
        max_grad_norm=1.0,
        warmup_ratio=ratio,
        lr_scheduler=kind,
        total_steps=S_TOTAL,
        **decay,
    )
    return optimizer, model


def _train(optimizer, model, updates, lrs=None, start=0):
    """Updates start+1 .. start+updates with a gradient that depends on the update."""
    for step in range(start, start + updates):
        model.w.grad = torch.full_like(model.w, 0.25 * (step % 3 + 1))
        if lrs is not None:
            lrs.append(optimizer.get_learning_rate())
        optimizer.step()


def _roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def _flat_keys(state):
    keys = set(state)
    keys |= {f"after_scheduler_dict.{k}" for k in state["after_scheduler_dict"]}
    return keys


class TestClosedForms(unittest.TestCase):
    def test_warmup_length(self):
        self.assertEqual(warmup_updates(RATIO, TOTAL), W)
        self.assertEqual(warmup_updates(S_RATIO, S_TOTAL), S_W)

    def test_scheduler_and_helpers_agree_bit_for_bit(self):
        for total, w, d, t in ((S_TOTAL, S_W, S_D, S_T), (TOTAL, W, D, T)):
            with self.subTest(total=total):
                optimizer = _adamw()
                scheduler = ConstantLinearDecayWarmupLR(
                    optimizer, total, w, decay_start=d, decay_end=t
                )
                lrs = _scheduler_lrs(scheduler, optimizer, t + 1)
                expected = [
                    constant_linear_decay_lr(u, BASE, w, d, t) for u in range(1, t + 2)
                ]
                self.assertEqual(lrs, expected)  # exact float equality
                constant_optimizer = _adamw()
                constant = ConstantWarmupLR(constant_optimizer, total, w)
                constant_lrs = _scheduler_lrs(constant, constant_optimizer, t + 1)
                self.assertEqual(
                    constant_lrs,
                    [constant_warmup_lr(u, BASE, w) for u in range(1, t + 2)],
                )
                # identical to constant through D, strictly decaying after
                self.assertEqual(lrs[:d], constant_lrs[:d])
                self.assertTrue(
                    all(a < b for a, b in zip(lrs[d + 1 : t], lrs[d : t - 1]))
                )

    def test_formula_endpoints(self):
        def lr(u):
            return constant_linear_decay_lr(u, BASE, W, D, T)

        self.assertEqual((lr(1), lr(W), lr(W + 1)), (BASE / W, BASE, BASE))
        self.assertEqual(lr(D), BASE)
        # exactly base: what a constant step-D checkpoint restores
        self.assertEqual(lr(D + 1), BASE)
        self.assertEqual(lr(T), BASE * (1 / (T - D)))
        self.assertAlmostEqual(lr(T), BASE / (T - D), delta=1e-15 * BASE)
        self.assertEqual(lr(T - 1), BASE * (2 / (T - D)))
        self.assertEqual(lr(14360), BASE * (2361 / 4720))
        self.assertEqual(lr(T + 1), 0.0)
        for u in (1, 7, 1999, 2000, 2001, 5000, 12000):
            self.assertEqual(lr(u), constant_warmup_lr(u, BASE, W))

    def test_helper_rejects_bad_windows(self):
        for w, d, t in (
            (2001, 1999, 16720),
            (2000, 12000, 12000),
            (2000, 12001, 12000),
            (2000, 12000.0, 16720),
        ):
            with self.subTest(window=(w, d, t)), self.assertRaises(ValueError):
                constant_linear_decay_lr(12001, BASE, w, d, t)

    def test_scheduler_rejects_end_past_total(self):
        with self.assertRaisesRegex(ValueError, "exceeds total_steps"):
            ConstantLinearDecayWarmupLR(_adamw(), 20, 4, decay_start=10, decay_end=21)

    def test_nested_scheduler_refuses_a_changed_window(self):
        optimizer = _adamw()
        scheduler = ConstantLinearDecayWarmupLR(
            optimizer, S_TOTAL, S_W, decay_start=S_D, decay_end=S_T
        )
        _scheduler_lrs(scheduler, optimizer, 6)
        state = _roundtrip(scheduler.state_dict())
        other = ConstantLinearDecayWarmupLR(
            _adamw(), S_TOTAL, S_W, decay_start=S_D, decay_end=S_T + 1
        )
        with self.assertRaisesRegex(ValueError, "lr_decay_end_update"):
            other.load_state_dict(state)

    def test_cosine_schedule_is_untouched(self):
        optimizer = _adamw()
        scheduler = CosineAnnealingWarmupLR(
            optimizer, total_steps=S_TOTAL, warmup_steps=S_W
        )
        lrs = _scheduler_lrs(scheduler, optimizer, 6)
        self.assertEqual(
            lrs[:S_W], [constant_warmup_lr(u, BASE, S_W) for u in range(1, S_W + 1)]
        )


class TestOptimizerBranch(unittest.TestCase):
    created = False

    @classmethod
    def setUpClass(cls):
        # BF16Optimizer.load_state_dict logs through print_on_rank0 (needs a group).
        import torch.distributed as dist

        if dist.is_available() and not dist.is_initialized():
            store = dist.FileStore(
                os.path.join(tempfile.mkdtemp(prefix="lr_decay_pg_"), "store"), 1
            )
            dist.init_process_group("gloo", store=store, rank=0, world_size=1)
            cls.created = True

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist

        if cls.created:
            dist.destroy_process_group()
            cls.created = False

    def test_cooldown_state_is_a_superset_of_constant(self):
        constant, _ = _optimizer("constant")
        cooldown, _ = _optimizer("constant_linear_decay")
        self.assertEqual(set(constant.state_dict()), STATE_KEYS)
        self.assertEqual(
            set(cooldown.state_dict()),
            STATE_KEYS | {"lr_decay_window", "lr_schedule_branch"},
        )
        self.assertLessEqual(
            _flat_keys(constant.scheduler.state_dict()),
            _flat_keys(cooldown.scheduler.state_dict()),
        )
        self.assertEqual(
            cooldown.state_dict()["lr_decay_window"],
            {
                "warmup_updates": S_W,
                "decay_start_update": S_D,
                "decay_end_update": S_T,
            },
        )
        self.assertIsNone(cooldown.state_dict()["lr_schedule_branch"])

    def test_constant_checkpoint_branches_into_the_cooldown_schedule(self):
        for branch_update in (0, 1, S_W - 1, S_W, S_W + 1, S_D - 1, S_D):
            with self.subTest(branch_update=branch_update):
                self._branch(branch_update)

    def _branch(self, branch_update):
        constant, model = _optimizer("constant")
        _train(constant, model, branch_update)
        state = _roundtrip(constant.state_dict())
        self.assertEqual(
            completed_updates_from_state(state["scheduler_state_dict"]), branch_update
        )

        cooldown, model2 = _optimizer("constant_linear_decay")
        model2.load_state_dict(model.state_dict())
        cooldown.load_state_dict(state)
        branch = {
            "from_type": "constant",
            "branch_update": branch_update,
            "decay_start_update": S_D,
            "decay_end_update": S_T,
        }
        self.assertEqual(cooldown.lr_schedule_branch, branch)
        self.assertEqual(cooldown.scheduler.completed_updates(), branch_update)
        lrs = []
        _train(cooldown, model2, S_T - branch_update, lrs, start=branch_update)
        self.assertEqual(
            lrs,
            [
                constant_linear_decay_lr(u, BASE, S_W, S_D, S_T)
                for u in range(branch_update + 1, S_T + 1)
            ],
        )
        self.assertEqual(
            cooldown.lr_schedule(),
            {
                "type": "constant_linear_decay",
                "base_lrs": [BASE],
                "warmup_updates": S_W,
                "decay_start_update": S_D,
                "decay_end_update": S_T,
                "branch": branch,
            },
        )

        # Same trajectory as a cooldown run that never branched (schedules agree <= D).
        straight, model3 = _optimizer("constant_linear_decay")
        _train(straight, model3, S_T)
        self.assertTrue(torch.equal(model3.w, model2.w))
        for a, b in zip(straight.fp32_params, cooldown.fp32_params):
            self.assertTrue(torch.equal(a, b))

        # The branch metadata survives a later cooldown -> cooldown resume.
        saved = _roundtrip(cooldown.state_dict())
        self.assertEqual(saved["lr_schedule_branch"], branch)
        resumed, _ = _optimizer("constant_linear_decay")
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.lr_schedule_branch, branch)
        self.assertEqual(resumed.get_learning_rate(), 0.0)  # update T + 1

    def test_cooldown_resume_mid_decay_continues_the_formula(self):
        cooldown, model = _optimizer("constant_linear_decay")
        _train(cooldown, model, 13)
        resumed, model2 = _optimizer("constant_linear_decay")
        resumed.load_state_dict(_roundtrip(cooldown.state_dict()))
        self.assertIsNone(resumed.lr_schedule_branch)  # fresh cooldown run
        model2.load_state_dict(model.state_dict())
        lrs = []
        _train(resumed, model2, S_T - 13, lrs, start=13)
        self.assertEqual(
            lrs,
            [
                constant_linear_decay_lr(u, BASE, S_W, S_D, S_T)
                for u in range(14, S_T + 1)
            ],
        )

    def test_branch_refused_past_the_decay_start(self):
        constant, model = _optimizer("constant")
        _train(constant, model, S_D + 1)
        cooldown, _ = _optimizer("constant_linear_decay")
        with self.assertRaisesRegex(ValueError, f"from update {S_D + 1}: it is past"):
            cooldown.load_state_dict(_roundtrip(constant.state_dict()))
        self.assertIsNone(cooldown.lr_schedule_branch)

    def test_branch_refused_when_warmup_differs(self):
        constant, model = _optimizer("constant", ratio=0.05)  # warmup 2
        _train(constant, model, 5)
        cooldown, _ = _optimizer("constant_linear_decay")
        with self.assertRaisesRegex(ValueError, "warmup"):
            cooldown.load_state_dict(_roundtrip(constant.state_dict()))

    def test_later_resume_refuses_a_changed_window(self):
        for change in (dict(d=S_D - 1), dict(t=S_T + 1), dict(d=S_D + 1, t=S_T + 2)):
            with self.subTest(**change):
                constant, model = _optimizer("constant")
                _train(constant, model, S_D - 2)
                cooldown, model2 = _optimizer("constant_linear_decay")
                cooldown.load_state_dict(_roundtrip(constant.state_dict()))
                _train(cooldown, model2, 3)
                saved = _roundtrip(cooldown.state_dict())
                changed, _ = _optimizer("constant_linear_decay", **change)
                with self.assertRaisesRegex(ValueError, "identical warmup/D/T"):
                    changed.load_state_dict(copy.deepcopy(saved))
                # Tampered branch metadata (window intact) is refused too.
                saved["lr_schedule_branch"]["decay_end_update"] = S_T + 5
                same, _ = _optimizer("constant_linear_decay")
                with self.assertRaisesRegex(ValueError, "lr_schedule_branch"):
                    same.load_state_dict(saved)

    def test_every_other_scheduler_change_is_refused(self):
        for saved, current in (
            ("constant", "cosine"),
            ("cosine", "constant"),
            ("cosine", "constant_linear_decay"),
            ("constant_linear_decay", "constant"),
            ("constant_linear_decay", "cosine"),
        ):
            with self.subTest(saved=saved, current=current):
                source, model = _optimizer(saved)
                _train(source, model, 3)
                target, _ = _optimizer(current)
                with self.assertRaisesRegex(ValueError, "lr_scheduler="):
                    target.load_state_dict(_roundtrip(source.state_dict()))

    def test_constant_to_constant_resume_is_unchanged(self):
        constant, model = _optimizer("constant")
        _train(constant, model, 7)
        resumed, model2 = _optimizer("constant")
        resumed.load_state_dict(_roundtrip(constant.state_dict()))
        self.assertIsNone(resumed.lr_schedule_branch)
        self.assertIsNone(resumed.lr_decay_window)
        lrs = []
        _train(resumed, model2, 5, lrs)
        self.assertEqual(lrs, [constant_warmup_lr(u, BASE, S_W) for u in range(8, 13)])
        self.assertEqual(set(resumed.state_dict()), set(constant.state_dict()))

    def test_optimizer_validates_decay_arguments(self):
        for kwargs, match in (
            (dict(lr_scheduler="constant_linear_decay"), "requires"),
            (
                dict(lr_scheduler="constant_linear_decay", lr_decay_start_update=10),
                "requires",
            ),
            (
                dict(
                    lr_scheduler="constant",
                    lr_decay_start_update=10,
                    lr_decay_end_update=17,
                ),
                "apply only",
            ),
            (dict(lr_scheduler="cosine", lr_decay_end_update=17), "apply only"),
            (
                dict(
                    lr_scheduler="constant_linear_decay",
                    lr_decay_start_update=3,
                    lr_decay_end_update=17,
                ),
                "warmup <= decay_start",
            ),
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, match):
                BF16Optimizer(
                    _Draft(),
                    lr=BASE,
                    warmup_ratio=S_RATIO,
                    total_steps=S_TOTAL,
                    **kwargs,
                )


class TestConfig(unittest.TestCase):
    MINIMAL = {
        "model": {"target_model_path": "target", "draft_model_config": "draft.json"},
        "data": {"hidden_states_path": "/features"},
    }
    COOLDOWN = dict(
        lr_scheduler="constant_linear_decay",
        lr_decay_start_update=D,
        lr_decay_end_update=T,
    )

    def _config(self, **training):
        from specforge.config import Config

        payload = copy.deepcopy(self.MINIMAL)
        payload["training"] = {
            "max_steps": T,
            "total_steps": TOTAL,
            "warmup_ratio": RATIO,
            "learning_rate": BASE,
            **training,
        }
        return Config.model_validate(payload)

    def test_cooldown_config_validates_and_round_trips(self):
        from specforge.config import Config

        cfg = self._config(**self.COOLDOWN)
        t = cfg.training
        self.assertEqual(
            (t.lr_scheduler, t.lr_decay_start_update, t.lr_decay_end_update),
            ("constant_linear_decay", D, T),
        )
        again = Config.model_validate(cfg.model_dump())
        self.assertEqual(
            (again.training.lr_decay_start_update, again.training.lr_decay_end_update),
            (D, T),
        )
        default = self._config().training
        self.assertEqual(
            (default.lr_decay_start_update, default.lr_decay_end_update), (None, None)
        )

    def test_window_edges_are_accepted(self):
        for training in (
            dict(lr_decay_start_update=W),  # D == warmup
            dict(lr_decay_end_update=D + 1),  # T == D + 1
            dict(lr_decay_start_update=T - 1),  # D == max_steps - 1
        ):
            with self.subTest(**training):
                self._config(**{**self.COOLDOWN, **training})

    def test_invalid_windows_are_rejected(self):
        from pydantic import ValidationError

        for training, match in (
            (dict(lr_scheduler="constant_linear_decay"), "requires"),
            (
                dict(lr_scheduler="constant_linear_decay", lr_decay_start_update=D),
                "requires",
            ),
            (
                dict(lr_scheduler="constant_linear_decay", lr_decay_end_update=T),
                "requires",
            ),
            (dict(self.COOLDOWN, lr_decay_start_update=W - 1), "warmup <="),
            (
                dict(self.COOLDOWN, lr_decay_end_update=D),
                "lr_decay_start_update < lr_decay_end_update",
            ),
            (dict(self.COOLDOWN, lr_decay_end_update=T + 1), "<= max_steps"),
            (dict(self.COOLDOWN, max_steps=None), "requires training.max_steps"),
            (dict(self.COOLDOWN, lr_decay_start_update=0), "greater than 0"),
            (
                dict(lr_decay_start_update=D, lr_decay_end_update=T),
                "apply only",
            ),
            (dict(lr_scheduler="cosine", lr_decay_start_update=D), "apply only"),
            (dict(lr_scheduler="linear"), "lr_scheduler"),
        ):
            with (
                self.subTest(**training),
                self.assertRaisesRegex(ValidationError, match),
            ):
                self._config(**training)

    def test_optimizer_factory_builds_the_configured_schedule(self):
        from specforge.training.assembly import _ConfiguredOptimizerFactory

        cooldown = _ConfiguredOptimizerFactory(self._config(**self.COOLDOWN))(_Draft())
        self.assertIsInstance(cooldown.scheduler, ConstantLinearDecayWarmupLR)
        self.assertEqual(
            cooldown.lr_schedule(),
            {
                "type": "constant_linear_decay",
                "base_lrs": [BASE],
                "warmup_updates": W,
                "decay_start_update": D,
                "decay_end_update": T,
                "branch": None,
            },
        )
        constant = _ConfiguredOptimizerFactory(self._config(lr_scheduler="constant"))(
            _Draft()
        )
        self.assertIs(type(constant.scheduler), ConstantWarmupLR)
        self.assertIsNone(constant.lr_decay_window)
        self.assertEqual(
            constant.lr_schedule(),
            {"type": "constant", "base_lrs": [BASE], "warmup_updates": W},
        )
        self.assertEqual(set(constant.state_dict()), STATE_KEYS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
