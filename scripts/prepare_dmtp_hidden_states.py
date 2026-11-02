"""Capture tokenized DMTP samples through the plugin into offline .ckpt files.

Input is JSONL with input_ids and loss_mask arrays (one document per line).
The external capture server and this client share MOONCAKE_* connection settings.
No SGLang implementation is imported by this client.
"""

import argparse
import json
import os
import uuid
from pathlib import Path

import torch
from transformers import AutoConfig

from specforge.algorithms.dmtp.data import normalize
from specforge.inference.adapters.server_capture import (
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask, SampleRef
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()
    target = AutoConfig.from_pretrained(args.target_model_path)
    target = getattr(target, "text_config", target)
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(
            "output directory must be empty to avoid mixing capture versions"
        )
    store = MooncakeFeatureStore(
        store_id="dmtp-offline-" + uuid.uuid4().hex,
        setup_kwargs={
            "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
            "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
            "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
            "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
            "rdma_devices": os.environ.get("MOONCAKE_DEVICE", ""),
            "global_segment_size": 1 << 30,
            "local_buffer_size": 1 << 30,
        },
    )
    schema = ServerCaptureSchema(
        aux_feature="hidden_states",
        last_hidden_feature=None,
        mtp_hidden_feature="mtp_hidden",
        passthrough=(("input_ids", "input_ids", ()), ("loss_mask", "loss_mask", ())),
    )
    adapter = SGLangServerCaptureAdapter(
        args.server_url,
        store,
        run_id=store.store_id,
        algorithm="dmtp",
        schema=schema,
        target_model_version=args.target_model_path,
        server_capture="plugin",
    )
    capture = CaptureConfig.from_strategy(
        {"input_ids", "loss_mask", "hidden_states", "mtp_hidden"},
        args.target_layer_ids,
        target_repr=None,
        target_hidden_size=target.hidden_size,
    )
    try:
        with open(args.input_path) as source:
            for index, line in enumerate(source):
                raw = json.loads(line)
                ids, mask = (
                    raw["input_ids"][: args.max_length],
                    raw["loss_mask"][: args.max_length],
                )
                if len(ids) != len(mask) or len(ids) < 4:
                    raise ValueError(
                        f"line {index+1}: input_ids/loss_mask must match and contain at least 4 tokens"
                    )
                task = PromptTask(
                    task_id=str(index),
                    run_id=store.store_id,
                    source_id=args.input_path,
                    payload={"input_ids": ids, "loss_mask": mask},
                    max_length=len(ids),
                )
                ref = adapter.produce_refs([task], capture=capture)[0]
                if not isinstance(ref, SampleRef):
                    raise RuntimeError(f"line {index+1}: {ref}")
                handle = None
                temporary = output / f"{index:08d}.tmp"
                try:
                    tensors, handle = store.get(ref)
                    sample = {
                        key: value.squeeze(0).cpu() for key, value in tensors.items()
                    }
                    sample["mtp_hidden_alignment"] = "anchor_post_norm_v1"
                    sample["mtp_hidden_version"] = torch.tensor(1, dtype=torch.int64)
                    sample["target_model_path"] = args.target_model_path
                    normalize(sample, args.max_length)
                    torch.save(sample, temporary)
                    temporary.replace(output / f"{index:08d}.ckpt")
                finally:
                    if handle is not None:
                        store.release(handle)
                    else:
                        store.abort(ref.sample_id, reason="offline capture failed")
                    temporary.unlink(missing_ok=True)
                print(f"saved {index+1} samples", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
