"""Opt-in native online integration fixture starting at the SampleRef boundary.

Run once with --prepare, then launch this file using torchrun. The public
frontend and actual retaining shared-directory store, inbox distributor, SQLite
ledger, ACKs and native DCP all run. The default shared_dir mode replaces only
the configured Mooncake store constructor. With --transport mooncake, real TCP
put/get is exercised against externally prepared endpoints in --mooncake-env.
Neither mode runs SGLang feature capture.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument(
        "--algorithm", choices=("dflash", "dflash2", "dspark"), default="dflash2"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--transport", choices=("shared_dir", "mooncake"), default="shared_dir"
    )
    parser.add_argument("--mooncake-env")
    args = parser.parse_args()
    if args.mooncake_env:
        os.environ.update(json.loads(Path(args.mooncake_env).read_text()))
    root, fixture = Path(args.root), Path(args.fixture)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["DISAGG_REF_CHANNEL"] = str(root / "refs.jsonl")
    os.environ["DISAGG_DB"] = str(root / "consumer.sqlite")
    os.environ["DISAGG_IDLE_TIMEOUT"] = "60"
    from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

    class FixtureStore(SharedDirFeatureStore):
        def close(self):
            pass

    def store_factory(cfg=None, *, retain_on_release=False):
        if args.transport == "mooncake":
            from specforge.config import load_config
            from specforge.training.disaggregated import _mooncake_store

            cfg = cfg or load_config(str(fixture / "train.yaml"))
            return _mooncake_store(cfg, retain_on_release=retain_on_release)
        return FixtureStore(
            str(root / "features"),
            store_id="native-online",
            retain_on_release=retain_on_release,
        )

    if args.prepare:
        store = store_factory()
        channel = StreamingRefChannel(str(root / "refs.jsonl"))
        generator = torch.Generator().manual_seed(35)
        for index in range(12):  # DP2 * accumulation2 * batch1 * 3 steps
            length = 8 + index % 3
            tensors = {
                "input_ids": torch.randint(0, 63, (1, length), generator=generator),
                "hidden_states": torch.randn(
                    1, length, 64, generator=generator, dtype=torch.bfloat16
                ),
                "target_last_hidden_states": torch.randn(
                    1, length, 32, generator=generator, dtype=torch.bfloat16
                ),
                "loss_mask": torch.ones(1, length, dtype=torch.long),
            }
            tensors["loss_mask"][:, : index % 2] = 0
            ref = store.put(
                tensors,
                sample_id=f"native-{index}",
                metadata={
                    "run_id": "native-online",
                    "strategy": "dspark" if args.algorithm == "dspark" else "dflash",
                    "num_tokens": length,
                    "target_repr": "hidden_state",
                },
            )
            channel.publish(ref)
        channel.close()
        from specforge.config import load_config
        from specforge.training.disaggregated import (
            _ONLINE_SCHEDULE_SUFFIX,
            _online_schedule_payload,
        )

        producer_config = load_config(str(fixture / "train.yaml"))
        (root / ("refs.jsonl" + _ONLINE_SCHEDULE_SUFFIX)).write_text(
            json.dumps(_online_schedule_payload(producer_config, num_prompts=12))
        )
        draft = json.loads((fixture / "draft.json").read_text())
        draft["architectures"] = [
            {
                "dflash": "DFlashDraftModel",
                "dflash2": "DFlash2DraftModel",
                "dspark": "DSparkDraftModel",
            }[args.algorithm]
        ]
        (root / "draft.json").write_text(json.dumps(draft))
        (root / "ready").touch()
        if args.transport == "mooncake":
            # Keep the producer-owned host segment alive through all consumer
            # attempts. The supervising test touches stop only after it is done.
            deadline = time.monotonic() + 7200
            while not (root / "stop").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Mooncake fixture producer exceeded two hours")
                time.sleep(0.5)
            store.close()
        return

    import specforge.training.disaggregated as disaggregated
    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.config import load_config
    from specforge.training.torchtitan.frontend import build_torchtitan_training_run

    if args.transport == "shared_dir":
        disaggregated._mooncake_store = store_factory
    cfg = load_config(str(fixture / "train.yaml"))
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"hidden_states_path": None}),
            "run_id": "native-online",
            "output_dir": str(root / "output"),
            "deployment": cfg.deployment.model_copy(update={"mode": "disaggregated"}),
            "model": cfg.model.model_copy(
                update={"draft_model_config": str(root / "draft.json")}
            ),
            "training": cfg.training.model_copy(
                update={
                    "role": "consumer",
                    "strategy": "dspark" if args.algorithm == "dspark" else "dflash",
                    "max_steps": args.steps,
                    "resume_from": args.resume,
                }
            ),
        }
    )
    algorithm = builtin_algorithm_registry().resolve(cfg.training.strategy)
    completed = build_torchtitan_training_run(cfg, algorithm=algorithm).run()
    expected_steps = min(args.steps or 3, 3)
    assert completed == expected_steps
    if int(os.environ.get("RANK", "0")) == 0:
        from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore

        ledger = SQLiteMetadataStore(str(root / "consumer.sqlite"))
        marker = ledger.durable_marker()
        assert marker["global_step"] == expected_steps and marker["optimizer_durable"]
        assert len(marker["acked"]) == 4 * expected_steps
        ledger.close()
        print(
            f"ONLINE_BOUNDARY_OK step={completed} acknowledged={4 * expected_steps}",
            flush=True,
        )


if __name__ == "__main__":
    main()
