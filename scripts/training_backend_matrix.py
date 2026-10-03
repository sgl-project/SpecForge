#!/usr/bin/env python3
"""Run and gate one immutable, matched FSDP/TorchTitan performance matrix.

The default invocation writes a plan only. --execute launches fresh processes;
it never modifies or copies a source checkout. Use a frozen source and driver.
Same-policy native/Graph tolerances are recorded before the first trial.
FSDP precision and Torch-version differences are reported separately, without
misrepresenting them as identical numerical implementations.
Graph timings failing the gate remain diagnostics; the separate FSDP/native
table is retained. Loss agreement is not a convergence/gradient proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark_training_backends import source_hashes
from training_benchmark_recipes import ARCHITECTURES, STOCK_CONFIGS

ALGORITHMS = ("dflash", "dflash2", "dspark")
CASES = {
    "fsdp213": ["--backend", "fsdp"],
    "fsdp214": ["--backend", "fsdp"],
    "titan-cuda": ["--backend", "torchtitan", "--compile", "--cuda-graphs"],
    "graph-full": [
        "--backend",
        "torchtitan",
        "--compile",
        "--cuda-graphs",
        "--titan-engine",
        "graph",
        "--graph-inductor",
        "full",
    ],
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_trials(
    python, source, driver, output, repeats, warmup, steps, gpus, *, python213
):
    """Rotate case order over repeats; all trials use the same GPU pair."""
    trials = []
    cases = list(CASES)
    for repeat in range(repeats):
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            offset = (repeat + algorithm_index) % len(cases)
            for case in cases[offset:] + cases[:offset]:
                name = f"{algorithm}-{case}-{repeat + 1}"
                command = [
                    str(python213 if case == "fsdp213" else python),
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={len(gpus.split(','))}",
                    str(driver / "scripts/benchmark_training_backends.py"),
                    "--specforge-root",
                    str(source),
                    "--algorithm",
                    algorithm,
                    "--warmup-steps",
                    str(warmup),
                    "--steps",
                    str(steps),
                    "--output",
                    str(output / f"{name}.json"),
                    *CASES[case],
                ]
                trials.append(
                    dict(
                        name=name,
                        algorithm=algorithm,
                        case=case,
                        repeat=repeat + 1,
                        command=command,
                    )
                )
    return trials


def loss_difference(
    reference, candidate, *, first_atol=None, trajectory_atol=None, rtol=None
):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Loss vectors must have the same nonzero length")
    if any(not math.isfinite(x) for x in [*reference, *candidate]):
        raise ValueError("Non-finite loss")
    errors = [abs(left - right) for left, right in zip(reference, candidate)]
    failures = [
        index + 1
        for index, (value, error) in enumerate(zip(reference, errors))
        if rtol is not None
        and error > (first_atol if index == 0 else trajectory_atol) + rtol * abs(value)
    ]
    return dict(
        passed=not failures if rtol is not None else None,
        first_step_absolute_error=errors[0],
        maximum_absolute_error=max(errors),
        maximum_relative_error=max(
            error / max(abs(value), 1e-30) for error, value in zip(errors, reference)
        ),
        failing_steps=failures,
    )


def identify_case(payload):
    runtime = payload["runtime"]
    identity = (
        payload["backend"],
        runtime["titan_engine"],
        runtime["compile"],
        runtime["cuda_graphs"],
        runtime["graph_inductor"],
    )
    expected = {
        ("fsdp", "trainer", False, False, None): "fsdp",
        ("torchtitan", "trainer", True, True, None): "titan-cuda",
        ("torchtitan", "graph", True, True, "full"): "graph-full",
    }
    if identity not in expected:
        raise ValueError(f"Unsupported benchmark runtime: {identity}")
    case = expected[identity]
    version = payload["versions"]["torch"]
    if case == "fsdp":
        if version.startswith("2.13."):
            return "fsdp213"
        if version.startswith("2.14."):
            return "fsdp214"
        raise ValueError(f"Unsupported FSDP baseline Torch version: {version}")
    if not version.startswith("2.14."):
        raise ValueError(f"Expected Torch2.14 for Titan; got {version}")
    return case


def validate_algorithm_identity(payload, algorithm):
    """Check recorded model identity instead of trusting the result filename."""
    architecture = payload["comparison_contract"].get("config", {}).get("architectures")
    if architecture != [ARCHITECTURES[algorithm]]:
        raise ValueError(
            f"Trial algorithm {algorithm} disagrees with recorded architecture: "
            f"{architecture}"
        )
    source = payload.get("recipe_source")
    # Controlled recipes share the DFlash JSON and override its architecture;
    # stock recipes instead use the selected algorithm's own source JSON.
    allowed_sources = {STOCK_CONFIGS["dflash"], STOCK_CONFIGS[algorithm]}
    if not isinstance(source, str) or Path(source).name not in allowed_sources:
        raise ValueError(f"Unexpected recipe source for {algorithm}: {source}")


def validate_and_summarize(records, tolerances, *, repeats):
    """Reject mixed snapshots/inputs and gate every repeat against its peers."""
    required = {
        (algorithm, case, repeat)
        for algorithm in ALGORITHMS
        for case in CASES
        for repeat in range(1, repeats + 1)
    }
    if set(records) != required:
        raise ValueError("Matrix has missing, duplicate, or unexpected trials")
    payloads = list(records.values())
    first = payloads[0]
    for payload in payloads:
        if any(
            key in payload for key in ("correctness_collector", "correctness_probe")
        ):
            raise ValueError(
                "Instrumented correctness probes are not performance trials"
            )
        for key in (
            "source_sha256",
            "benchmark_sha256",
            "recipe_helpers_sha256",
            "device",
        ):
            if payload[key] != first[key]:
                raise ValueError(f"Mixed benchmark identity: {key}")
        for key in (
            "kernel_environment",
            "compute_dtype",
            "adam_state_dtype",
            "float32_matmul_precision",
            "allow_tf32_matmul",
            "allow_tf32_cudnn",
            "deterministic_algorithms",
            "visible_devices",
        ):
            if payload["runtime"][key] != first["runtime"][key]:
                raise ValueError(f"Mixed execution setting: {key}")
    gates, observations, rows = [], [], []
    for family in (("fsdp213",), ("fsdp214", "titan-cuda", "graph-full")):
        family_payloads = [
            payload for (_, case, _), payload in records.items() if case in family
        ]
        for payload in family_payloads:
            if (
                payload["versions"] != family_payloads[0]["versions"]
                or payload["cuda"] != family_payloads[0]["cuda"]
            ):
                raise ValueError("Mixed environment within one Torch version family")
    if len({payload["versions"]["transformers"] for payload in payloads}) != 1:
        raise ValueError(
            "Transformers versions must match across both Torch environments"
        )
    for algorithm in ALGORITHMS:
        reference_contract = records[(algorithm, "fsdp213", 1)]["comparison_contract"]
        reference_source = records[(algorithm, "fsdp213", 1)].get("recipe_source")
        count = reference_contract["warmup_steps"] + reference_contract["steps"]
        for case in CASES:
            for repeat in range(1, repeats + 1):
                payload = records[(algorithm, case, repeat)]
                validate_algorithm_identity(payload, algorithm)
                if identify_case(payload) != case:
                    raise ValueError("Trial name disagrees with its runtime")
                if payload["recipe_source"] != reference_source:
                    raise ValueError(f"Mixed recipe sources for {algorithm}")
                if payload["comparison_contract"] != reference_contract:
                    raise ValueError(
                        f"Unequal model/input contract: {algorithm}/{case}"
                    )
                if len(payload["losses_including_warmup"]) != count:
                    raise ValueError("Incomplete loss trajectory")
                seconds = payload["step_seconds_max_rank"]
                if len(seconds) != reference_contract["steps"] or any(
                    not math.isfinite(value) or value <= 0 for value in seconds
                ):
                    raise ValueError("Incomplete or invalid timing vector")
        for repeat in range(1, repeats + 1):
            for reference_case, candidate_case in (("titan-cuda", "graph-full"),):
                gate = loss_difference(
                    records[(algorithm, reference_case, repeat)][
                        "losses_including_warmup"
                    ],
                    records[(algorithm, candidate_case, repeat)][
                        "losses_including_warmup"
                    ],
                    **tolerances,
                )
                gates.append(
                    dict(
                        algorithm=algorithm,
                        repeat=repeat,
                        reference=reference_case,
                        candidate=candidate_case,
                        **gate,
                    )
                )
        for repeat in range(1, repeats + 1):
            for reference_case, candidate_case in (
                ("fsdp213", "fsdp214"),
                ("fsdp214", "titan-cuda"),
            ):
                observations.append(
                    dict(
                        algorithm=algorithm,
                        repeat=repeat,
                        reference=reference_case,
                        candidate=candidate_case,
                        **loss_difference(
                            records[(algorithm, reference_case, repeat)][
                                "losses_including_warmup"
                            ],
                            records[(algorithm, candidate_case, repeat)][
                                "losses_including_warmup"
                            ],
                        ),
                    )
                )
        baseline_means = [
            statistics.mean(
                records[(algorithm, "fsdp213", repeat)]["step_seconds_max_rank"]
            )
            for repeat in range(1, repeats + 1)
        ]
        baseline214_means = [
            statistics.mean(
                records[(algorithm, "fsdp214", repeat)]["step_seconds_max_rank"]
            )
            for repeat in range(1, repeats + 1)
        ]
        for case in CASES:
            runs = [
                records[(algorithm, case, repeat)] for repeat in range(1, repeats + 1)
            ]
            means = [statistics.mean(run["step_seconds_max_rank"]) for run in runs]
            median = statistics.median(means)
            rows.append(
                dict(
                    algorithm=algorithm,
                    case=case,
                    trial_mean_seconds=means,
                    median_trial_mean_seconds=median,
                    paired_speedups_vs_fsdp213=[
                        base / value for base, value in zip(baseline_means, means)
                    ],
                    median_paired_speedup_vs_fsdp213=statistics.median(
                        base / value for base, value in zip(baseline_means, means)
                    ),
                    median_paired_speedup_vs_fsdp214=statistics.median(
                        base / value for base, value in zip(baseline214_means, means)
                    ),
                    peak_allocated_gib=max(
                        run["peak_allocated_bytes_max_rank"] for run in runs
                    )
                    / 2**30,
                    runtime=runs[0]["runtime"],
                )
            )
    passed = all(gate["passed"] for gate in gates)
    return dict(
        schema="specforge-matched-backends-v1",
        passed=passed,
        tolerances=tolerances,
        source_sha256=first["source_sha256"],
        benchmark_sha256=first["benchmark_sha256"],
        recipe_helpers_sha256=first["recipe_helpers_sha256"],
        gates=gates,
        fsdp_precision_observations=observations,
        performance_rows=(
            rows if passed else [row for row in rows if row["case"] != "graph-full"]
        ),
        graph_diagnostic_rows=(
            [] if passed else [row for row in rows if row["case"] == "graph-full"]
        ),
        graph_gate_passed=passed,
        environment_versions={
            case: records[(ALGORITHMS[0], case, 1)]["versions"] for case in CASES
        },
        limitation="Loss agreement does not establish gradient equivalence, convergence, or serving quality.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specforge-root", required=True, type=Path)
    parser.add_argument(
        "--driver-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--python213", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--first-atol", type=float, default=1e-5)
    parser.add_argument("--trajectory-atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if min(args.repeats, args.warmup_steps, args.steps) < 1:
        parser.error("repeats, warmup-steps and steps must be positive")
    if args.execute and args.analyze_only:
        parser.error("--execute and --analyze-only are mutually exclusive")
    gpu_ids = args.gpus.split(",")
    if (
        len(gpu_ids) != 2
        or len(set(gpu_ids)) != 2
        or not all(value.isdigit() for value in gpu_ids)
    ):
        parser.error("This matched DP2 matrix requires two distinct numeric GPU IDs")
    tolerances = {
        key: getattr(args, key) for key in ("first_atol", "trajectory_atol", "rtol")
    }
    if any(not math.isfinite(value) or value < 0 for value in tolerances.values()):
        parser.error("Tolerances must be finite and nonnegative")
    source, driver, output = (
        value.resolve()
        for value in (args.specforge_root, args.driver_root, args.output)
    )
    trials = plan_trials(
        args.python,
        source,
        driver,
        output,
        args.repeats,
        args.warmup_steps,
        args.steps,
        args.gpus,
        python213=args.python213,
    )
    hashes = source_hashes(source)
    driver_hashes = {
        name: file_hash(driver / "scripts" / name)
        for name in (
            "benchmark_training_backends.py",
            "training_benchmark_recipes.py",
            "training_backend_matrix.py",
        )
    }
    plan = dict(
        trials=trials,
        tolerances=tolerances,
        source_sha256=hashes,
        driver_sha256=driver_hashes,
        gpus=args.gpus,
    )
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "matrix-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(
            "Existing plan has different source, tolerances, or arguments; use a new directory"
        )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute and not args.analyze_only:
        print(
            f"Planned {len(trials)} fresh-process trials in {plan_path}; no GPU work launched"
        )
        return
    if args.execute and any(
        (output / f"{trial['name']}.json").exists() for trial in trials
    ):
        raise ValueError(
            "Refusing to pool previous results; use a new output directory"
        )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "PYTHONPATH": str(source),
        "OMP_NUM_THREADS": "1",
        "SPECFORGE_DFLASH_FUSED_HEAD": "1",
        "SPECFORGE_DFLASH2_FUSED_CONV": "1",
    }
    launches, records = [], {}
    for trial in trials:
        result_path = output / f"{trial['name']}.json"
        if args.execute:
            if result_path.exists():
                raise ValueError(
                    "Refusing to pool previous results; use a new output directory"
                )
            if source_hashes(source) != hashes or any(
                file_hash(driver / "scripts" / name) != digest
                for name, digest in driver_hashes.items()
            ):
                raise ValueError("Source or benchmark driver changed after planning")
            started = time.time()
            print(json.dumps({"start": trial["name"], "time": started}), flush=True)
            with (output / f"{trial['name']}.log").open("w") as log:
                completed = subprocess.run(
                    trial["command"],
                    cwd=source,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            launches.append(
                dict(
                    **trial,
                    started=started,
                    wall_seconds=time.time() - started,
                    exit_code=completed.returncode,
                    environment={
                        key: environment[key]
                        for key in (
                            "CUDA_VISIBLE_DEVICES",
                            "OMP_NUM_THREADS",
                            "SPECFORGE_DFLASH_FUSED_HEAD",
                            "SPECFORGE_DFLASH2_FUSED_CONV",
                        )
                    },
                )
            )
            (output / "launch-manifest.json").write_text(
                json.dumps(launches, indent=2) + "\n"
            )
            completed.check_returncode()
        payload = json.loads(result_path.read_text())
        if (
            payload["comparison_contract"]["warmup_steps"] != args.warmup_steps
            or payload["comparison_contract"]["steps"] != args.steps
        ):
            raise ValueError("Result duration does not match the recorded plan")
        if payload["source_sha256"] != hashes or any(
            payload[key] != driver_hashes[name]
            for key, name in (
                ("benchmark_sha256", "benchmark_training_backends.py"),
                ("recipe_helpers_sha256", "training_benchmark_recipes.py"),
            )
        ):
            raise ValueError("Result does not match the frozen source/driver plan")
        records[(trial["algorithm"], trial["case"], trial["repeat"])] = payload
    report = validate_and_summarize(records, tolerances, repeats=args.repeats)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "trials": len(records),
                "report": str(output / "comparison.json"),
            }
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(
            "Graph numerical gate failed; Graph timings are diagnostic only. FSDP/native rows retained."
        )


if __name__ == "__main__":
    main()
