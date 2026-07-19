"""Architecture guards for the post-consolidation package layout."""

from __future__ import annotations

import ast
import json
import re
import tokenize
import tomllib
import unittest
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# These modules were temporary move-only compatibility shims. Keeping the list
# explicit makes an accidental reintroduction fail without importing torch or
# optional inference backends.
REMOVED_MODULE_FILES = (
    "specforge/modeling/target/base.py",
    "specforge/modeling/target/factory.py",
    "specforge/modeling/target/dflash_target_model.py",
    "specforge/modeling/target/eagle3_target_model.py",
    "specforge/modeling/target/sglang_backend/__init__.py",
    "specforge/runtime/inference/capture.py",
    "specforge/runtime/inference/dflash_adapter.py",
    "specforge/runtime/inference/sglang_adapter.py",
    "specforge/runtime/training/__init__.py",
    "specforge/args.py",
    "specforge/inference/adapters/dflash.py",
    "specforge/inference/adapters/eagle3.py",
    "specforge/inference/adapters/policy.py",
    "specforge/runtime/data_plane/local_rollout_stream.py",
    "specforge/inference/target_engine/capture_policy.py",
    "specforge/inference/target_engine/dflash_target_model.py",
    "specforge/inference/target_engine/eagle3_target_model.py",
    "specforge/modeling/utils.py",
)

REMOVED_TRAINING_ENTRY_FILES = (
    "scripts/train_eagle3.py",
    "scripts/train_eagle3_dataflow.py",
    "scripts/train_dflash.py",
    "scripts/train_domino.py",
    "scripts/train_peagle.py",
    "examples/disagg/run_disagg_eagle3.py",
    "examples/disagg/run_disagg_dflash.py",
    "examples/disagg/run_disagg_domino.py",
    "examples/disagg/run_disagg_dspark.py",
)

REMOVED_SUPERSEDED_PATH_TESTS = (
    "tests/test_runtime/test_disagg_online.py",
    "tests/test_runtime/test_disagg_online_interleave.py",
)

# Deleted test filenames are historical capability labels, not paths we expect
# to restore. Each label maps to the checked-in canonical-runtime gates that now
# own that behavior. The architecture test below makes this mapping executable:
# cleanup cannot silently turn a real correctness gate into prose-only intent.
CAPABILITY_TEST_REPLACEMENTS = {
    "tests/test_runtime/test_backpressure.py": (
        "tests/test_runtime/test_flow_control.py",
        "tests/test_runtime/test_disagg_multiserver.py",
    ),
    "tests/test_runtime/test_dflash_launch.py": (
        "tests/test_runtime/test_dflash_offline_launch.py",
    ),
    "tests/test_runtime/test_disagg_online_shared_plane.py": (
        "tests/test_runtime/test_disagg_online_shared_plane.py",
        "tests/test_runtime/test_ref_distributor.py",
    ),
    "tests/test_runtime/test_recovery.py": (
        "tests/test_runtime/test_recovery.py",
        "tests/test_runtime/test_ref_distributor.py",
    ),
    "tests/test_runtime/test_resharding.py": (
        "tests/test_runtime/test_ref_distributor.py",
    ),
    "tests/test_scripts/test_compact_teacher_integration.py": (
        "tests/test_runtime/test_compact_teacher_strategy.py",
    ),
    "tests/test_scripts/test_disagg_dflash_mask_token.py": (
        "tests/test_runtime/test_model_assembly_utils.py",
    ),
    "tests/test_scripts/test_train_eagle3.py": (
        "tests/test_runtime/test_single_entry.py",
        "tests/test_runtime/test_offline_launch_fsdp.py",
    ),
    "tests/test_scripts/test_train_eagle3_cache_key.py": (
        "tests/test_runtime/test_offline_vocab_mapping.py",
    ),
    "tests/test_scripts/test_train_eagle3_optimizer.py": (
        "tests/test_runtime/test_domain_trainer.py",
        "tests/test_runtime/test_trainer.py",
    ),
    "tests/test_scripts/test_train_peagle_resume.py": (
        "tests/test_runtime/test_checkpoint_resume.py",
        "tests/test_runtime/test_peagle_strategy.py",
    ),
    "tests/test_utils/test_compact_teacher.py": (
        "tests/test_runtime/test_compact_teacher_strategy.py",
    ),
}

# High-value operational examples removed by the hard cutover must retain a
# concrete, checked-in destination.  This is deliberately narrower than a
# filename-for-filename migration of every legacy model shell: the complete
# model catalog is guarded by the typed YAML reachability tests, while this map
# protects disaggregated topology, performance, and gate behavior that cannot
# be inferred from a draft config alone.
OPERATIONAL_EXAMPLE_REPLACEMENTS = {
    "examples/disagg/PERF_FINDINGS.md": (
        "docs/benchmarks/domino-disaggregated-performance.md",
    ),
    "examples/disagg/run_qwen2.5_7b_eagle3_disagg.sh": (
        "examples/configs/qwen2.5-7b-eagle3-offline-disaggregated.yaml",
        "examples/disagg/run_offline_2node.sh",
        "docs/benchmarks/eagle3-disaggregated-parity.md",
    ),
    "examples/disagg/run_qwen3.6_27b_dflash_disagg.sh": (
        "examples/configs/qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml",
        "examples/configs/qwen3.6-27b-dflash-disaggregated.yaml",
    ),
    "examples/disagg/run_qwen3.6_27b_dflash_disagg_multiserver.sh": (
        "examples/configs/qwen3.6-27b-dflash-multiserver-disaggregated.yaml",
    ),
    "examples/disagg/run_qwen3_8b_dflash_disagg_1srv_dp7.sh": (
        "examples/configs/qwen3-8b-dflash-1server-dp7-disaggregated.yaml",
    ),
    "examples/disagg/run_qwen3_8b_domino_disagg_1srv_dp7.sh": (
        "examples/configs/qwen3-8b-domino-1server-dp7-disaggregated.yaml",
        "docs/benchmarks/domino-disaggregated-performance.md",
    ),
    "examples/disagg/run_qwen3_8b_domino_disagg_multiserver.sh": (
        "examples/configs/qwen3-8b-domino-multiserver-disaggregated.yaml",
    ),
    "examples/disagg/run_domino_dflash_serving_gate.sh": (
        "scripts/gates/run_dflash_serving_gate.sh",
    ),
    "examples/disagg/run_domino_disagg_overfit_gate.sh": (
        "scripts/gates/run_disaggregated_overfit_gate.sh",
    ),
}

REMOVED_PACKAGE_DIRECTORIES = (
    "specforge/modeling/target/sglang_backend",
    "specforge/modeling/target/custom_backend",
    "specforge/inference/target_engine",
    "tests/test_modeling/test_target/test_custom_backend",
    "tests/test_modeling/test_target/test_sglang_backend",
    "specforge/runtime/inference",
    "specforge/runtime/training",
)

REMOVED_MODULE_PREFIXES = (
    "specforge.modeling.target.base",
    "specforge.modeling.target.factory",
    "specforge.modeling.target.dflash_target_model",
    "specforge.modeling.target.eagle3_target_model",
    "specforge.modeling.target.sglang_backend",
    "specforge.modeling.target.custom_backend",
    "specforge.runtime.inference",
    "specforge.runtime.training",
    "specforge.args",
    "specforge.inference.adapters.dflash",
    "specforge.inference.adapters.eagle3",
    "specforge.inference.adapters.policy",
    "specforge.inference.target_engine",
    "specforge.runtime.data_plane.local_rollout_stream",
    "specforge.inference.target_engine.capture_policy",
    "specforge.inference.target_engine.dflash_target_model",
    "specforge.inference.target_engine.eagle3_target_model",
    "specforge.modeling.utils",
    "scripts.train_eagle3",
    "scripts.train_eagle3_dataflow",
    "scripts.train_dflash",
    "scripts.train_domino",
    "scripts.train_peagle",
    "train_eagle3",
    "train_eagle3_dataflow",
    "train_dflash",
    "train_domino",
    "train_peagle",
)

SOURCE_ROOTS = (
    REPO_ROOT / "specforge",
    REPO_ROOT / "scripts",
    REPO_ROOT / "examples",
    REPO_ROOT / "tests",
)

PRODUCTION_ROOTS = (
    REPO_ROOT / "specforge",
    REPO_ROOT / "scripts",
    REPO_ROOT / "examples",
)

LEGACY_ENTRY_GLOBS = (
    "scripts/train_*.py",
    "examples/run_*online*.sh",
    "examples/disagg/run_disagg_*.py",
)

LEGACY_API_NAMES = {
    "AutoEagle3DraftModel",
    "AutoDistributedTargetModel",
    "CAPTURE_POLICIES",
    "CapturePolicy",
    "CaptureSpec",
    "CustomEagle3TargetModel",
    "DFlashAdapter",
    "DFlashTargetModel",
    "Eagle3TargetModel",
    "HFEagle3TargetModel",
    "SGLangAdapter",
    "SGLangEagle3TargetModel",
    "build_disagg_eagle3_runtime",
    "build_disagg_online_eagle3_runtime",
    "build_disagg_online_runtime",
    "build_offline_eagle3_controller",
    "build_offline_eagle3_runtime",
    "build_online_eagle3_runtime",
    "build_online_runtime",
    "get_dflash_target_model",
    "get_eagle3_target_model",
    "register_capture_policy",
    "resolve_capture_policy",
    "run_disagg_online_interleaved",
    "_is_closed",
    "_remap_legacy_draft_head_keys",
    "TrainLease",
    "cap_train_lease",
    "dp_partition",
    "dataflow_colocated",
    "enqueue_offline_refs",
    "fail_refs",
    "lease_train_refs",
    "max_train_lease",
    "note_trainer_starved",
    "partition_key",
    "train_lease",
    "build_control_plane_for_mode",
}

CANONICAL_LAUNCH_EXPORTS = {
    "build_disagg_offline_runtime",
    "build_disagg_online_consumer",
    "build_disagg_online_producer",
    "build_offline_runtime",
}

DRAFT_MODEL_BUILDERS = CANONICAL_LAUNCH_EXPORTS - {
    "build_disagg_online_producer",
}

CANONICAL_DRAFT_CONFIGS = {
    "glm-5.2-dspark.json",
    "inkling-dspark.json",
    "llama3-8B-eagle3.json",
    "qwen3-4b-dspark.json",
    "qwen3-8b-dspark.json",
    "qwen3-8b-dflash.json",
    "qwen3-8b-domino.json",
    "qwen3-8b-eagle3.json",
    "qwen3-8b-peagle.json",
}

CANONICAL_DATASET_PRESETS = ("ultrachat", "sharegpt")

DOC_ONLY_PACKAGE_INITIALIZERS = (
    "specforge/__init__.py",
    "specforge/core/__init__.py",
    "specforge/data/__init__.py",
    "specforge/inference/__init__.py",
    "specforge/modeling/__init__.py",
    "specforge/modeling/target/__init__.py",
)


def _is_removed_module(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in REMOVED_MODULE_PREFIXES
    )


def _imported_modules(path: Path) -> Iterator[tuple[int, str]]:
    with tokenize.open(path) as source_file:
        tree = ast.parse(source_file.read(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module


def _module_tree(path: Path) -> ast.Module:
    with tokenize.open(path) as source_file:
        return ast.parse(source_file.read(), filename=str(path))


def _literal_all(path: Path) -> set[str]:
    return set(_literal_assignment(path, "__all__"))


def _literal_assignment(path: Path, name: str):
    for node in _module_tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(
        f"{path.relative_to(REPO_ROOT)} has no literal assignment for {name}"
    )


class TestPackageArchitecture(unittest.TestCase):
    def test_package_exposes_one_training_cli(self):
        with open(REPO_ROOT / "pyproject.toml", "rb") as project_file:
            project = tomllib.load(project_file)
        self.assertEqual(
            {"specforge": "specforge.cli:main"},
            project["project"]["scripts"],
        )

    def test_removed_move_shims_stay_deleted(self):
        present = [
            relative_path
            for relative_path in REMOVED_MODULE_FILES + REMOVED_PACKAGE_DIRECTORIES
            if (REPO_ROOT / relative_path).exists()
        ]
        self.assertEqual([], present, f"old-path shims reintroduced: {present}")

    def test_legacy_training_entries_stay_deleted(self):
        present = [
            relative_path
            for relative_path in REMOVED_TRAINING_ENTRY_FILES
            if (REPO_ROOT / relative_path).exists()
        ]
        self.assertEqual(
            [], present, f"legacy training entries reintroduced: {present}"
        )

        globbed = sorted(
            str(path.relative_to(REPO_ROOT))
            for pattern in LEGACY_ENTRY_GLOBS
            for path in REPO_ROOT.glob(pattern)
        )
        self.assertEqual([], globbed, f"legacy entry pattern reintroduced: {globbed}")

        superseded_tests = [
            path
            for path in REMOVED_SUPERSEDED_PATH_TESTS
            if (REPO_ROOT / path).exists()
        ]
        self.assertEqual(
            [],
            superseded_tests,
            f"tests for superseded path topology reintroduced: {superseded_tests}",
        )

    def test_deleted_capabilities_have_checked_in_canonical_replacements(self):
        overlap = set(CAPABILITY_TEST_REPLACEMENTS) & set(REMOVED_SUPERSEDED_PATH_TESTS)
        self.assertEqual(set(), overlap)

        invalid = {}
        missing = {}
        for capability, replacements in CAPABILITY_TEST_REPLACEMENTS.items():
            bad_paths = [
                path
                for path in replacements
                if not path.startswith("tests/test_runtime/test_")
                or not path.endswith(".py")
            ]
            if bad_paths:
                invalid[capability] = bad_paths
            absent = [path for path in replacements if not (REPO_ROOT / path).is_file()]
            if absent:
                missing[capability] = absent
        self.assertEqual({}, invalid, f"invalid canonical test paths: {invalid}")
        self.assertEqual({}, missing, f"missing canonical capability gates: {missing}")

    def test_removed_operational_examples_have_explicit_destinations(self):
        restored_legacy = [
            path
            for path in OPERATIONAL_EXAMPLE_REPLACEMENTS
            if (REPO_ROOT / path).exists()
        ]
        missing = {
            path: [
                replacement
                for replacement in replacements
                if not (REPO_ROOT / replacement).is_file()
            ]
            for path, replacements in OPERATIONAL_EXAMPLE_REPLACEMENTS.items()
        }
        missing = {path: paths for path, paths in missing.items() if paths}

        self.assertEqual([], restored_legacy)
        self.assertEqual({}, missing, f"missing operational migrations: {missing}")

    def test_repository_does_not_import_removed_modules(self):
        violations = []
        for source_root in SOURCE_ROOTS:
            for path in sorted(source_root.rglob("*.py")):
                for line_number, module in _imported_modules(path):
                    if _is_removed_module(module):
                        relative_path = path.relative_to(REPO_ROOT)
                        violations.append(f"{relative_path}:{line_number}: {module}")

        self.assertEqual(
            [],
            violations,
            "imports must use specforge.inference or specforge.training:\n"
            + "\n".join(violations),
        )

    def test_production_does_not_reference_legacy_api_symbols(self):
        violations = []
        for source_root in PRODUCTION_ROOTS:
            for path in sorted(source_root.rglob("*.py")):
                for node in ast.walk(_module_tree(path)):
                    name = None
                    if isinstance(node, ast.Name):
                        name = node.id
                    elif isinstance(node, ast.Attribute):
                        name = node.attr
                    elif isinstance(node, ast.alias):
                        name = node.name.rsplit(".", 1)[-1]
                    elif isinstance(node, ast.arg):
                        name = node.arg
                    elif isinstance(node, ast.keyword):
                        name = node.arg
                    if name in LEGACY_API_NAMES:
                        relative_path = path.relative_to(REPO_ROOT)
                        violations.append(f"{relative_path}:{node.lineno}: {name}")
        self.assertEqual(
            [], violations, "legacy API references:\n" + "\n".join(violations)
        )

    def test_launch_exports_only_canonical_topology_builders(self):
        self.assertEqual(
            CANONICAL_LAUNCH_EXPORTS,
            _literal_all(REPO_ROOT / "specforge" / "launch.py"),
        )

    def test_public_training_lifecycle_has_one_surface(self):
        trainer_tree = _module_tree(REPO_ROOT / "specforge" / "training" / "trainer.py")
        trainer_class = next(
            node
            for node in trainer_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Trainer"
        )
        methods = {
            node.name: node
            for node in trainer_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("from_strategy_name", methods)
        for name in ("fit", "save_checkpoint"):
            args = methods[name].args
            self.assertEqual(["self"], [arg.arg for arg in args.args], name)
            self.assertEqual([], args.kwonlyargs, name)
            self.assertIsNone(args.vararg, name)
            self.assertIsNone(args.kwarg, name)

        init_attrs = {
            node.attr
            for node in ast.walk(methods["__init__"])
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }
        self.assertTrue({"_controller", "_loader"}.issubset(init_attrs))
        self.assertTrue({"controller", "loader"}.isdisjoint(init_attrs))

        assembly_tree = _module_tree(
            REPO_ROOT / "specforge" / "training" / "assembly.py"
        )
        run_class = next(
            node
            for node in assembly_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TrainingRun"
        )
        run_fields = {
            node.target.id
            for node in run_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertNotIn("loader", run_fields)
        run_method = next(
            node
            for node in run_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        fit_calls = [
            node
            for node in ast.walk(run_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        ]
        self.assertEqual(1, len(fit_calls))
        self.assertEqual([], fit_calls[0].args)
        self.assertEqual([], fit_calls[0].keywords)

        launch_tree = _module_tree(REPO_ROOT / "specforge" / "launch.py")
        launch_functions = {
            node.name: node
            for node in launch_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assembler_returns = [
            node
            for node in ast.walk(launch_functions["_assemble_trainer"])
            if isinstance(node, ast.Return)
        ]
        self.assertEqual(1, len(assembler_returns))
        self.assertIsInstance(assembler_returns[0].value, ast.Name)
        self.assertEqual("trainer", assembler_returns[0].value.id)

        trainer_builders = CANONICAL_LAUNCH_EXPORTS - {"build_disagg_online_producer"}
        for name in trainer_builders:
            returns = [
                node
                for node in ast.walk(launch_functions[name])
                if isinstance(node, ast.Return) and node.value is not None
            ]
            self.assertTrue(returns, name)
            for node in returns:
                self.assertNotIsInstance(node.value, ast.Tuple, name)
                self.assertFalse(
                    isinstance(node.value, ast.Name) and node.value.id == "loader",
                    name,
                )
                self.assertFalse(
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr in {"controller", "loader"},
                    name,
                )
        self.assertNotIn(
            "run_interleaved", (REPO_ROOT / "specforge" / "launch.py").read_text()
        )

        for relative_path in (
            "specforge/training/assembly.py",
            "specforge/training/disaggregated.py",
        ):
            for node in ast.walk(_module_tree(REPO_ROOT / relative_path)):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "trainer"
                ):
                    self.assertNotIn(
                        node.attr,
                        {"controller", "loader"},
                        relative_path,
                    )
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "trainer"
                ):
                    continue
                if node.func.attr == "fit":
                    self.assertEqual([], node.args, relative_path)
                    self.assertEqual([], node.keywords, relative_path)
                self.assertNotEqual("save_checkpoint", node.func.attr, relative_path)

    def test_server_only_builders_have_one_stream_path(self):
        functions = {
            node.name: node
            for node in _module_tree(REPO_ROOT / "specforge" / "launch.py").body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        consumer_parameters = {
            arg.arg
            for arg in functions["build_disagg_online_consumer"].args.args
            + functions["build_disagg_online_consumer"].args.kwonlyargs
        }
        self.assertIn("resume_from", consumer_parameters)
        self.assertTrue({"resume", "num_epochs"}.isdisjoint(consumer_parameters))

        offline_parameters = {
            arg.arg
            for arg in functions["build_offline_runtime"].args.args
            + functions["build_offline_runtime"].args.kwonlyargs
        }
        self.assertTrue(
            {"deployment_mode", "metadata_db_path"}.isdisjoint(offline_parameters)
        )

        producer_parameters = {
            arg.arg
            for arg in functions["build_disagg_online_producer"].args.args
            + functions["build_disagg_online_producer"].args.kwonlyargs
        }
        self.assertTrue(
            {"metadata_store", "metadata_db_path"}.isdisjoint(producer_parameters)
        )

        consumer = functions["build_disagg_online_consumer"]
        called = [
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(consumer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        ]
        self.assertEqual(1, called.count("RefDistributor"))
        self.assertIn("DPAckController", called)
        self.assertNotIn("DataFlowController", called)
        self.assertIn("publish_consumer_quantum", called)
        queue_call = next(
            node
            for node in ast.walk(consumer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "StreamingRefQueue"
        )
        self.assertIsInstance(queue_call.args[0], ast.Name)
        self.assertEqual("inbox", queue_call.args[0].id)

        producer = functions["build_disagg_online_producer"]
        controller_call = next(
            node
            for node in ast.walk(producer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DataFlowController"
        )
        keywords = {keyword.arg: keyword.value for keyword in controller_call.keywords}
        self.assertIs(keywords["enable_sample_queue"].value, False)
        self.assertEqual("NoOpMetadataStore", keywords["metadata_store"].func.id)

        for name in ("build_offline_runtime", "build_disagg_offline_runtime"):
            controller_call = next(
                node
                for node in ast.walk(functions[name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "DataFlowController"
            )
            controller_keywords = {
                keyword.arg: keyword.value for keyword in controller_call.keywords
            }
            self.assertIs(controller_keywords["enable_sample_queue"].value, False, name)

    def test_aggregate_packages_stay_doc_only(self):
        for relative_path in DOC_ONLY_PACKAGE_INITIALIZERS:
            body = _module_tree(REPO_ROOT / relative_path).body
            self.assertEqual(1, len(body), relative_path)
            self.assertIsInstance(body[0], ast.Expr, relative_path)
            self.assertIsInstance(body[0].value, ast.Constant, relative_path)
            self.assertIsInstance(body[0].value.value, str, relative_path)

    def test_data_regeneration_dependency_is_explicit(self):
        with open(REPO_ROOT / "pyproject.toml", "rb") as project_file:
            extras = tomllib.load(project_file)["project"]["optional-dependencies"]
        self.assertEqual(["openai"], extras["data"])

    def test_launch_builders_use_generic_draft_model_parameter(self):
        functions = {
            node.name: node
            for node in _module_tree(REPO_ROOT / "specforge" / "launch.py").body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in CANONICAL_LAUNCH_EXPORTS:
            parameters = {
                arg.arg
                for arg in functions[name].args.args + functions[name].args.kwonlyargs
            }
            self.assertNotIn("eagle3_model", parameters, name)
            if name in DRAFT_MODEL_BUILDERS:
                self.assertIn("draft_model", parameters, name)

    def test_data_preparation_exposes_only_canonical_presets(self):
        script_path = REPO_ROOT / "scripts" / "prepare_data.py"
        self.assertTrue(
            set(CANONICAL_DATASET_PRESETS).issubset(
                _literal_assignment(script_path, "SUPPORTED_DATASETS")
            )
        )

    def test_offline_capture_uses_only_the_dedicated_local_capture(self):
        source = (REPO_ROOT / "scripts" / "prepare_hidden_states.py").read_text()
        self.assertNotIn("--enable-aux-hidden-states", source)
        self.assertNotIn("--aux-hidden-states-layers", source)
        self.assertNotIn("Eagle3TargetModel", source)
        self.assertNotIn("get_target_engine", source)
        self.assertNotIn("return_last_hidden_states=True", source)
        self.assertNotIn("return_logits=False", source)
        self.assertIn("load_offline_eagle3_capture", source)

    def test_only_canonical_draft_configs_are_checked_in(self):
        present = {path.name for path in (REPO_ROOT / "configs").glob("*.json")}
        self.assertTrue(CANONICAL_DRAFT_CONFIGS.issubset(present))

    def test_dspark_configs_are_qwen3_gqa_only(self):
        dspark_configs = {}
        for path in sorted((REPO_ROOT / "configs").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "DSparkDraftModel" in payload.get("architectures", []):
                dspark_configs[path.name] = payload

        self.assertEqual(
            set(dspark_configs),
            {
                "glm-5.2-dspark.json",
                "inkling-dspark.json",
                "qwen3-4b-dspark.json",
                "qwen3-8b-dspark.json",
            },
        )
        for name, payload in dspark_configs.items():
            with self.subTest(config=name):
                self.assertEqual(payload["model_type"], "qwen3")
                self.assertEqual(payload["dflash_config"]["attention_mode"], "gqa")
                self.assertLess(
                    payload["num_key_value_heads"],
                    payload["num_attention_heads"],
                )
                self.assertEqual(
                    payload["num_attention_heads"] % payload["num_key_value_heads"],
                    0,
                )

    def test_examples_and_scripts_do_not_bypass_the_cli(self):
        direct_imports = []
        for root in (REPO_ROOT / "scripts", REPO_ROOT / "examples"):
            for path in sorted(root.rglob("*.py")):
                for line_number, module in _imported_modules(path):
                    if module == "specforge.launch" or module.startswith(
                        "specforge.training"
                    ):
                        direct_imports.append(
                            f"{path.relative_to(REPO_ROOT)}:{line_number}: {module}"
                        )
        self.assertEqual(
            [], direct_imports, "CLI bypass imports:\n" + "\n".join(direct_imports)
        )

        allowed = {
            Path("examples/disagg/run_online.sh"),
            Path("examples/disagg/run_offline.sh"),
            Path("examples/disagg/run_offline_2node.sh"),
            Path("examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh"),
        }
        bypasses = []
        train_command = re.compile(r"\btrain\s+(?:--config|-c)\b")
        for root in (REPO_ROOT / "scripts", REPO_ROOT / "examples"):
            for path in sorted(root.rglob("*.sh")):
                relative = path.relative_to(REPO_ROOT)
                if train_command.search(path.read_text()) and relative not in allowed:
                    bypasses.append(str(relative))
        self.assertEqual(
            [], bypasses, f"non-canonical shell training entries: {bypasses}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
