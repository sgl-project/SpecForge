"""CPU/Gloo regressions for length-scheduling setup and resume error fences."""

from __future__ import annotations

import json
import multiprocessing
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from specforge.runtime.contracts import SampleRef

WORLD_SIZE = 2
GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()


def _ref(index, work_dir, *, num_tokens=0):
    return SampleRef(
        sample_id=str(index),
        run_id="length-fence-test",
        source_task_id=None,
        feature_store_uri=f"file://{work_dir}/sample-{index}.ckpt",
        feature_keys={"input_ids": "input_ids"},
        feature_specs={},
        strategy="dflash",
        num_tokens=num_tokens,
    )


def _offline_case(rank, scenario, work_dir):
    from specforge.data import offline_lengths

    refs = [_ref(0, work_dir, num_tokens=7), _ref(1, work_dir, num_tokens=11)]
    if scenario == "count_mismatch" and rank == 1:
        refs = refs[:1]
    elif scenario == "reordered_manifest" and rank == 1:
        refs.reverse()
    elif scenario == "load_error":
        refs = [_ref(0, work_dir)]

    with (
        mock.patch.object(
            offline_lengths, "_index_lengths", wraps=offline_lengths._index_lengths
        ) as index,
        mock.patch(
            "specforge.runtime.data_plane.feature_store.load_feature_file",
            side_effect=OSError("injected rank-zero feature read failure"),
        ) as load,
    ):
        try:
            offline_lengths.ensure_offline_lengths(
                refs, cache_dir=str(Path(work_dir) / "length-cache")
            )
        except Exception as exc:
            result = {"type": type(exc).__name__, "message": str(exc)}
        else:
            result = {"type": "success", "message": ""}
        result.update(index_calls=index.call_count, load_calls=load.call_count)
        return result


def _offline_heartbeat_case(rank, scenario, work_dir):
    from specforge.data import offline_lengths
    from specforge.runtime.data_plane import feature_store

    refs = [_ref(index, work_dir) for index in range(4)]
    messages = []
    original_broadcast = dist.broadcast_object_list
    original_load = feature_store.load_feature_file

    def record_broadcast(payload, *args, **kwargs):
        original_broadcast(payload, *args, **kwargs)
        # Record received contents, including on nonzero ranks, so the test
        # catches dropping, duplicating, or mistaking progress for a result.
        messages.append(dict(payload[0]))

    def load_feature(path):
        if scenario == "heartbeat_error" and path.endswith("sample-3.ckpt"):
            raise OSError("injected failure after multiple indexing heartbeats")
        return original_load(path)

    with (
        mock.patch.object(offline_lengths, "_INDEX_HEARTBEAT_SECONDS", 0.0),
        mock.patch.object(dist, "broadcast_object_list", side_effect=record_broadcast),
        mock.patch.object(
            feature_store, "load_feature_file", side_effect=load_feature
        ) as load,
    ):
        try:
            indexed = offline_lengths.ensure_offline_lengths(
                refs, cache_dir=str(Path(work_dir) / "length-cache")
            )
        except Exception as exc:
            result = {"type": type(exc).__name__, "message": str(exc)}
        else:
            result = {
                "type": "success",
                "sample_ids": [ref.sample_id for ref in indexed],
                "lengths": [ref.num_tokens for ref in indexed],
            }
        result.update(messages=messages, load_calls=load.call_count)
        return result


def _online_kwargs(work_dir):
    from specforge.algorithms.builtin import builtin_algorithm_registry

    return dict(
        algorithm=builtin_algorithm_registry().resolve("eagle3"),
        feature_store=SimpleNamespace(retain_on_release=True, abort=mock.Mock()),
        channel=SimpleNamespace(path=str(Path(work_dir) / "refs.jsonl")),
        draft_model=None,
        optimizer_factory=None,
        run_id="length-fence-test",
        output_dir=str(work_dir),
        metadata_db_path=str(Path(work_dir) / "ledger.sqlite"),
        resume_from=str(Path(work_dir) / "checkpoint"),
        async_ack=False,
    )


def _online_case(rank, scenario, work_dir):
    from specforge.launch import build_disagg_online_consumer
    from specforge.training.checkpoint import CheckpointManager

    kwargs = _online_kwargs(work_dir)
    kwargs.update(dp_rank=rank, dp_size=WORLD_SIZE, length_aware_scheduling=True)
    if scenario == "invalid_batch":
        # Rank zero fails before checkpoint inspection, while rank one reads
        # its policy. Both must enter the same outer preflight collective.
        kwargs["batch_size"] = 2 if rank == 0 else 1
    elif scenario == "resume_mismatch":
        kwargs["length_aware_scheduling"] = rank == 1

    with (
        mock.patch.object(
            CheckpointManager,
            "read_resume_state",
            side_effect=AssertionError("preflight entered checkpoint collectives"),
        ) as read_resume,
        mock.patch(
            "specforge.launch._resolve_metadata_store",
            side_effect=AssertionError("preflight reached ledger setup"),
        ) as resolve_store,
        mock.patch(
            "specforge.launch._assemble_trainer",
            side_effect=AssertionError("preflight reached trainer setup"),
        ) as assemble,
        mock.patch.object(
            dist, "all_gather_object", wraps=dist.all_gather_object
        ) as gather,
    ):
        try:
            build_disagg_online_consumer(**kwargs)
        except Exception as exc:
            result = {"type": type(exc).__name__, "message": str(exc)}
        else:
            result = {"type": "success", "message": ""}
        result.update(
            resume_calls=read_resume.call_count,
            ledger_calls=resolve_store.call_count,
            trainer_calls=assemble.call_count,
            gather_calls=gather.call_count,
            abort_calls=kwargs["feature_store"].abort.call_count,
        )
        return result


def _distributed_worker(rank, scenario, work_dir):
    result = {}
    try:
        dist.init_process_group(
            "gloo",
            init_method=(Path(work_dir) / "gloo-init").resolve().as_uri(),
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=20),
        )
        if scenario in {"invalid_batch", "resume_mismatch"}:
            result = _online_case(rank, scenario, work_dir)
        elif scenario in {"heartbeat_success", "heartbeat_error"}:
            result = _offline_heartbeat_case(rank, scenario, work_dir)
        else:
            result = _offline_case(rank, scenario, work_dir)
    except BaseException:
        result = {"worker_error": traceback.format_exc()}
    finally:
        (Path(work_dir) / f"rank-{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_ranks(scenario, work_dir):
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(
            target=_distributed_worker, args=(rank, scenario, str(work_dir))
        )
        for rank in range(WORLD_SIZE)
    ]
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + 45
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    stuck = [rank for rank, worker in enumerate(workers) if worker.is_alive()]
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
            if worker.is_alive():
                worker.kill()
                worker.join(5)
    assert not stuck, f"length scheduling setup deadlocked on ranks {stuck}"
    assert [worker.exitcode for worker in workers] == [0] * WORLD_SIZE
    results = []
    for rank in range(WORLD_SIZE):
        result = json.loads((work_dir / f"rank-{rank}.json").read_text())
        assert "worker_error" not in result, result.get("worker_error")
        results.append(result)
    return results


@pytest.mark.skipif(not GLOO_AVAILABLE, reason="requires CPU/Gloo")
@pytest.mark.parametrize("scenario", ["count_mismatch", "reordered_manifest"])
def test_offline_manifest_disagreement_rejects_every_rank_before_indexing(
    tmp_path, scenario
):
    results = _run_two_ranks(scenario, tmp_path)
    for result in results:
        assert result["type"] == "ValueError"
        assert (
            "offline feature lists differ between training ranks" in result["message"]
        )
        assert result["index_calls"] == 0
        assert result["load_calls"] == 0
    assert results[0]["message"] == results[1]["message"]
    assert not (tmp_path / "length-cache").exists()


@pytest.mark.skipif(not GLOO_AVAILABLE, reason="requires CPU/Gloo")
def test_offline_rank_zero_read_error_reaches_every_rank(tmp_path):
    # The stat succeeds; only rank zero attempts the failing tensor load.
    (tmp_path / "sample-0.ckpt").write_bytes(b"feature fixture")
    results = _run_two_ranks("load_error", tmp_path)
    for result in results:
        assert result["type"] == "RuntimeError"
        assert "offline length indexing failed" in result["message"]
        assert "OSError: injected rank-zero feature read failure" in result["message"]
    assert results[0]["message"] == results[1]["message"]
    assert [result["index_calls"] for result in results] == [1, 0]
    assert [result["load_calls"] for result in results] == [1, 0]


@pytest.mark.skipif(not GLOO_AVAILABLE, reason="requires CPU/Gloo")
@pytest.mark.parametrize("scenario", ["heartbeat_success", "heartbeat_error"])
def test_offline_multiple_heartbeats_are_followed_by_final_result(tmp_path, scenario):
    lengths = [7, 11, 13, 17]
    for index, length in enumerate(lengths):
        torch.save(
            {"input_ids": torch.arange(length)}, tmp_path / f"sample-{index}.ckpt"
        )
    results = _run_two_ranks(scenario, tmp_path)
    for result in results:
        assert result["messages"][:-1] == [{"progress": True}] * len(lengths)
        if scenario == "heartbeat_success":
            assert result["type"] == "success"
            assert result["sample_ids"] == ["0", "1", "2", "3"]
            assert result["lengths"] == lengths
            assert result["messages"][-1] == {"lengths": lengths}
        else:
            assert result["type"] == "RuntimeError"
            error = "OSError: injected failure after multiple indexing heartbeats"
            assert result["messages"][-1] == {"error": error}
            assert result["message"] == f"offline length indexing failed: {error}"
    assert results[0]["messages"] == results[1]["messages"]
    assert [result["load_calls"] for result in results] == [4, 0]


def _write_checkpoint(work_dir, policy):
    from specforge.training.checkpoint import STATE_FILE

    checkpoint = work_dir / "checkpoint"
    checkpoint.mkdir()
    state = {} if policy is None else {"online_length_aware_scheduling": policy}
    torch.save(state, checkpoint / STATE_FILE)


@pytest.mark.skipif(not GLOO_AVAILABLE, reason="requires CPU/Gloo")
@pytest.mark.parametrize("scenario", ["invalid_batch", "resume_mismatch"])
def test_online_mixed_rank_preflight_has_one_collective_and_no_ledger_setup(
    tmp_path, scenario
):
    _write_checkpoint(tmp_path, True)
    results = _run_two_ranks(scenario, tmp_path)
    expected = (
        "EAGLE3 with batch_size=1"
        if scenario == "invalid_batch"
        else "original online_length_aware_scheduling policy"
    )
    for result in results:
        assert result["type"] == "RuntimeError"
        assert "online consumer preflight failed" in result["message"]
        assert expected in result["message"]
        assert result["resume_calls"] == 0
        assert result["ledger_calls"] == 0
        assert result["trainer_calls"] == 0
        assert result["gather_calls"] == 1
        assert result["abort_calls"] == 0
    assert results[0]["message"] == results[1]["message"]
    assert not (tmp_path / "ledger.sqlite").exists()


@pytest.mark.parametrize("saved,current", [(False, True), (True, False), (None, True)])
def test_online_resume_cannot_change_scheduling_before_ledger_mutation(
    tmp_path, saved, current
):
    from specforge.launch import build_disagg_online_consumer
    from specforge.training.checkpoint import CheckpointManager

    _write_checkpoint(tmp_path, saved)
    kwargs = _online_kwargs(tmp_path)
    kwargs["length_aware_scheduling"] = current
    with (
        mock.patch.object(
            CheckpointManager,
            "read_resume_state",
            side_effect=AssertionError("preflight entered checkpoint collectives"),
        ) as read_resume,
        mock.patch(
            "specforge.launch._resolve_metadata_store",
            side_effect=AssertionError("preflight reached ledger setup"),
        ) as resolve_store,
        mock.patch(
            "specforge.launch._assemble_trainer",
            side_effect=AssertionError("preflight reached trainer setup"),
        ) as assemble,
        pytest.raises(
            ValueError, match="original online_length_aware_scheduling policy"
        ),
    ):
        build_disagg_online_consumer(**kwargs)
    read_resume.assert_not_called()
    resolve_store.assert_not_called()
    assemble.assert_not_called()
    kwargs["feature_store"].abort.assert_not_called()
    assert not (tmp_path / "ledger.sqlite").exists()
