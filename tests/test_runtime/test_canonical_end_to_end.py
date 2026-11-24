"""End-to-end canonical runs through the public entry points, on tiny synthetic data.

Every architecture trains from offline features, resumes, seeds a new run and is
exported; one run goes through ``specforge train`` itself, and the bundled
recipes are planned against synthetic target metadata.
"""

import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import yaml
from accelerate.utils import set_seed
from safetensors.torch import load_file

from specforge.application import build_application_run
from specforge.cli import main
from specforge.config import load_config
from specforge.config.draft.codec import encode_config
from specforge.distributed import Topology, destroy_distributed
from specforge.loading import load_draft
from specforge.training.checkpoint import CheckpointManager
from specforge.utils import get_device_type
from tests.test_algorithms.test_objectives import training_batch
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_modeling.test_canonical_dflash2 import dflash2_config
from tests.test_modeling.test_canonical_dmtp import model_config as dmtp_config
from tests.test_modeling.test_canonical_domino import model_config as domino_config
from tests.test_modeling.test_canonical_dspark import model_config as dspark_config
from tests.test_modeling.test_canonical_dspine import model_config as dspine_config
from tests.test_modeling.test_canonical_eagle3 import model_config as eagle_config
from tests.test_modeling.test_canonical_peagle import model_config as peagle_config
from tests.test_runtime.test_canonical_training import target_directory

ROOT = Path(__file__).resolve().parents[2]
CUT, STEPS = 2, 4
LEARNING_RATE = 1e-3
# torchrun's rendezvous variables; an earlier in-process test may have left them set.
DISTRIBUTED_ENV = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE",
)
# The pinned SGLang revision serves these; every other architecture must refuse.
SGLANG_SERVED = {
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "DSparkDraftModel",
    "DominoDraftModel",
    "Eagle3DraftModel",
}
# What a resumed run must reproduce: weights, optimizer and RNG, data position.
TRAINING_STATE = (
    "draft_state_dict",
    "backend",
    "global_step",
    "epoch",
    "epoch_batch",
    "epoch_samples",
)


def drafts():
    """One tiny draft per architecture, bound to the target as the presets are."""
    return [
        draft_config(),
        dflash2_config(bound=True),
        dspark_config(bound=True),
        domino_config(bound=True),
        dspine_config(),
        dmtp_config(bound=True),
        mtp_config(),
        eagle_config(bound=True, target_width=32),
        peagle_config(target_width=32),
    ]


def write_features(features, draft, records):
    """Write fresh offline records from the current CPU RNG stream."""
    features.mkdir()
    for index in range(records):
        record = {
            name: value[0].detach() for name, value in training_batch(draft).items()
        }
        torch.save(record, features / f"{index}.ckpt")


def write_run(root, draft, *, records=3):
    """Write a synthetic target, offline features and the run YAML; return its path."""
    torch.manual_seed(0)
    target = target_directory(root)
    features = root / "features"
    write_features(features, draft, records)
    raw = dict(
        draft=encode_config(draft),
        model=dict(
            target_model_path=str(target),
            torch_dtype="float32",
            # Reference kernels, so CPU and accelerator hosts run the same math.
            use_liger_kernel=False,
            use_fused_conv=False,
        ),
        data=dict(
            hidden_states_path=str(features),
            dataloader_num_workers=0,
            cache_dir=str(root / "cache"),
        ),
        training=dict(
            attention_backend="eager",
            fsdp_sharding="NO_SHARD",
            learning_rate=LEARNING_RATE,
            num_epochs=2,
            max_steps=STEPS,
            # Pinned: a run stopped early must keep the uninterrupted schedule.
            total_steps=2 * STEPS,
            loss=dict(fused=False),
            # Fewer anchors than supervised positions: each step draws from the RNG.
            objective_options=(
                {"num_anchors": 2} if draft.spec_config.protocol else {}
            ),
        ),
        output_dir=str(root / "reference"),
    )
    if draft.architecture == "MTPDraftModel":
        # MTP has no implicit initialization; an empty section selects scratch.
        raw["initialization"] = {}
    path = root / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@contextlib.contextmanager
def process_group(root):
    """Single-rank group for the trainer's DDP wrapper; reuse one already up."""
    if dist.is_initialized():
        yield
        return
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def current_topology():
    """The one-rank job of the default group that is up."""
    return Topology(device_type=get_device_type(), world_group=dist.group.WORLD)


def build(config, *overrides):
    """Assemble a run as the ``specforge train`` worker does, in the current group."""
    cfg = load_config(str(config), list(overrides))
    set_seed(cfg.training.seed)
    return build_application_run(cfg, topology=current_topology())


def train(config, *overrides):
    return build(config, *overrides).run()


def checkpoint_dir(output, step):
    return Path(output) / f"specforge-run-step{step}"


def training_state(directory):
    """Read a checkpoint on CPU through the trainer's own resume reader."""
    state = CheckpointManager.read_resume_state(str(directory))
    return {name: state[name] for name in TRAINING_STATE}


def saved_weights(directory):
    return training_state(directory)["draft_state_dict"]


def export(root, name, *options):
    """Run ``specforge export`` on the reference run's last checkpoint."""
    source = checkpoint_dir(root / "reference", STEPS)
    paths = ["--checkpoint", str(source), "--target-model", str(root / "target")]
    return main(["export", *paths, "--output-dir", str(root / name), *options])


def recomputed_tolerance():
    """Tolerance for state that two runs computed independently: none on CPU.

    Accelerator kernels may differ in the last bits. A tenth of one Adam step
    still separates that from lost optimizer, RNG or data state, which moves
    weights by whole steps.
    """
    return 0.0 if get_device_type() == "cpu" else LEARNING_RATE / 10


def assert_same_state(test, actual, expected, *, atol=0.0, where="state"):
    """Compare nested state: float tensors within ``atol``, everything else exactly."""
    if isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys(), where)
        for key, value in expected.items():
            assert_same_state(
                test, actual[key], value, atol=atol, where=f"{where}.{key}"
            )
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(len(actual), len(expected), where)
        for index, value in enumerate(expected):
            assert_same_state(
                test, actual[index], value, atol=atol, where=f"{where}[{index}]"
            )
    elif torch.is_tensor(expected):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=atol if expected.is_floating_point() else 0,
            msg=lambda message: f"{where}: {message}",
        )
    else:
        test.assertEqual(actual, expected, where)


def assert_same_tensors(test, actual, expected):
    """Match tensors by value: a serving export may rename weights, never alter them."""
    unmatched = list(actual.values())
    for name, tensor in expected.items():
        index = next(
            (
                index
                for index, other in enumerate(unmatched)
                if other.dtype == tensor.dtype
                and other.shape == tensor.shape
                and torch.equal(other, tensor)
            ),
            None,
        )
        test.assertIsNotNone(index, f"{name} is missing from the export")
        del unmatched[index]
    test.assertEqual(len(unmatched), 0, "the export adds tensors")


def write_target_metadata(directory, draft):
    """Write the target config.json a draft payload expects; plans read no weights."""
    features = draft["spec_config"]["target_features"]
    config = {
        "hidden_size": features["hidden_size"],
        "vocab_size": draft["vocab_size"],
        # Captured layers must exist; a final-hidden draft needs at least two.
        "num_hidden_layers": max(features.get("target_layer_ids", [0])) + 2,
    }
    transform = draft.get("target_logit_transform") or {}
    if transform.get("type") == "mup":
        config["logits_mup_width_multiplier"] = transform["multiplier"]
    (directory / "config.json").write_text(json.dumps(config))


class CanonicalLifecycleTest(unittest.TestCase):
    """Each architecture, through what ``specforge train`` and ``export`` execute."""

    @classmethod
    def setUpClass(cls):
        root = Path(cls.enterClassContext(TemporaryDirectory()))
        cls.enterClassContext(process_group(root))

    def test_training_updates_the_draft_and_saves_a_loadable_checkpoint(self):
        for draft in drafts():
            with (
                self.subTest(architecture=draft.architecture),
                TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                run = build(write_run(root, draft))
                # Saved before the first update: the initial weights.
                initial = saved_weights(run.trainer.save_checkpoint().checkpoint_uri)
                self.assertEqual(run.run(), STEPS)
                final = checkpoint_dir(root / "reference", STEPS)
                trained = saved_weights(final)
                changed = [
                    name
                    for name, tensor in trained.items()
                    if not torch.equal(tensor, initial[name])
                ]
                self.assertTrue(changed)
                model = load_draft(final, target=root / "target")
                assert_same_state(self, model.state_dict(), trained)

    def test_resumed_run_matches_the_uninterrupted_run(self):
        for draft in drafts():
            with (
                self.subTest(architecture=draft.architecture),
                TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                config = write_run(root, draft)
                self.assertEqual(train(config), STEPS)
                output = f"output_dir={root / 'interrupted'}"
                self.assertEqual(
                    train(config, output, f"training.max_steps={CUT}"), CUT
                )
                cut = checkpoint_dir(root / "interrupted", CUT)
                # A recipe's initialization (MTP's scratch marker) excludes resume.
                resume = (output, f"resume={cut}", "initialization=null")
                self.assertEqual(train(config, *resume), STEPS)
                assert_same_state(
                    self,
                    training_state(checkpoint_dir(root / "interrupted", STEPS)),
                    training_state(checkpoint_dir(root / "reference", STEPS)),
                    atol=recomputed_tolerance(),
                )

    def test_initialization_checkpoint_starts_from_the_saved_weights(self):
        for draft in drafts():
            with (
                self.subTest(architecture=draft.architecture),
                TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                config = write_run(root, draft)
                train(config)
                source = checkpoint_dir(root / "reference", STEPS)
                run = build(
                    config,
                    # Replaces the section: a dotted override cannot create one.
                    "initialization=" + json.dumps({"checkpoint": str(source)}),
                    f"output_dir={root / 'imported'}",
                    # Another seed: weights the import misses cannot match by chance.
                    "training.seed=7",
                    "training.max_steps=1",
                )
                # Saved before the first update: what the import loaded.
                imported = saved_weights(run.trainer.save_checkpoint().checkpoint_uri)
                assert_same_state(self, imported, saved_weights(source))
                # Weights only: the new run starts its own step count.
                self.assertEqual(run.run(), 1)

    def test_export_round_trips_the_checkpoint_weights(self):
        for draft in drafts():
            with (
                self.subTest(architecture=draft.architecture),
                TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                train(write_run(root, draft))
                saved = saved_weights(checkpoint_dir(root / "reference", STEPS))
                self.assertEqual(export(root, "hf"), 0)
                model = load_draft(root / "hf", target=root / "target")
                assert_same_state(self, model.state_dict(), saved)
                convert = ("serving", "--converter", "sglang")
                if draft.architecture in SGLANG_SERVED:
                    self.assertEqual(export(root, *convert), 0)
                    weights = load_file(root / "serving" / "model.safetensors")
                    assert_same_tensors(self, weights, saved)
                else:
                    with self.assertRaisesRegex(ValueError, "SGLang"):
                        export(root, *convert)
                    self.assertFalse((root / "serving").exists())


class TrainCommandTest(unittest.TestCase):
    def test_train_command_matches_the_in_process_run(self):
        """``specforge train`` stops, resumes by override and equals ``train``."""
        if dist.is_initialized():
            self.skipTest("specforge train creates the default process group itself")
        # A worker that fails only aborts its groups; do not leak them to other tests.
        self.addCleanup(
            lambda: dist.is_initialized() and destroy_distributed(current_topology())
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = write_run(root, drafts()[0])
            output = f"output_dir={root / 'command'}"
            cut = checkpoint_dir(root / "command", CUT)
            command = ["train", "-c", str(config), output]
            with patch.dict(os.environ):
                for name in DISTRIBUTED_ENV:
                    os.environ.pop(name, None)
                self.assertEqual(main(command + [f"training.max_steps={CUT}"]), 0)
                self.assertTrue(cut.is_dir())
                # The lifecycle tests resume through `resume`; this is its alias.
                resume = f"training.resume_from={cut}"
                self.assertEqual(main(command + [resume]), 0)
            with process_group(root):
                self.assertEqual(train(config), STEPS)
            assert_same_state(
                self,
                training_state(checkpoint_dir(root / "command", STEPS)),
                training_state(checkpoint_dir(root / "reference", STEPS)),
                atol=recomputed_tolerance(),
            )


class BundledRecipeTest(unittest.TestCase):
    def test_recipes_plan_against_synthetic_target_metadata(self):
        """``specforge train --plan`` resolves the draft, objective and launch."""
        families = set()
        for path in sorted((ROOT / "examples/configs").rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text())
            deployment = raw["deployment"]
            if "managed_local" in (deployment.get("disaggregated") or {}):
                # Planning a managed stack probes this host's capture servers and HCAs.
                continue
            with (
                self.subTest(recipe=str(path.relative_to(ROOT))),
                TemporaryDirectory() as target,
            ):
                draft = raw["draft"]
                overrides = [f"model.target_model_path={target}"]
                if isinstance(draft, str):
                    # Recipes name presets relative to the repository root.
                    overrides.append(f"draft={ROOT / draft}")
                    draft = yaml.safe_load((ROOT / draft).read_text())
                write_target_metadata(Path(target), draft)
                rendered = io.StringIO()
                # An inherited torchrun environment would plan this process as a worker.
                with (
                    patch.dict(os.environ, {}, clear=True),
                    contextlib.redirect_stdout(rendered),
                ):
                    status = main(["train", "-c", str(path), "--plan", *overrides])
                self.assertEqual(status, 0)
                plan = json.loads(rendered.getvalue())
                colocated = deployment["mode"] == "local_colocated"
                self.assertEqual(plan["role"], "all" if colocated else "both")
                families.update(draft["architectures"])
        # DSpine and DMTP ship no recipe; every other architecture must stay planned.
        expected = {draft.architecture for draft in drafts()}
        self.assertLessEqual(
            expected - {"DSpineDraftModel", "DMTPDraftModel"}, families
        )
