"""
In this script, we benchmark the attention mechanism of the model.
export TP=8
export MODEL_PATH=/root/huggingface_cache/Qwen2.5-7B-Instruct/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun \
--nproc_per_node=$TP \
--master_port=29500 \
offline_eagle_data_sglang.py \
--model-path $MODEL_PATH \
--mem-frac=0.95 --tp-size $TP \
--enable-return-hidden-states \
--outdir outdir0
"""
import os
import torch
import argparse
import logging
import random
import dataclasses
from typing import Tuple

import numpy as np
from datasets import concatenate_datasets
from offline_eagle_data import generate_data, split_range, DATASET_INFO
import multiprocessing

from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.utils import (
    DeepEPMode,
    configure_logger,
    get_bool_env_var,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
)
from sglang.bench_one_batch import load_model, BenchArgs

def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        Scheduler.prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_cpu_group=model_runner.tp_group.cpu_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            enable_two_batch_overlap=model_runner.server_args.enable_two_batch_overlap,
            enable_deepep_moe=model_runner.server_args.enable_deepep_moe,
            deepep_mode=DeepEPMode[model_runner.server_args.deepep_mode],
        )

@torch.no_grad
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
    logits_output, _ = model_runner.forward(forward_batch)
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    return logits_output.hidden_states
        
def sgl_generate(dataset, server_args, port_args, bench_args, tp_rank):
    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    if bench_args.profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        )
        profiler.start()
    # Prepare inputs for warm up
    sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
    reqs = [] # TODO: prepare inputs
    hidden_states_cpu = []
    for idx, row in enumerate(dataset):
        req = Req(
            rid=str(idx),
            origin_input_text="",
            origin_input_ids=row["input_ids"].tolist(),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        print(f"{tp_rank}\t{len(req.fill_ids)}")
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
        if len(reqs) == 8:
            hidden_states = extend(reqs, model_runner)
            hidden_states_cpu.append(hidden_states.to("cpu", non_blocking=True))
            # rank_print(hidden_states.shape)
            reqs = []
    torch.cuda.synchronize()
    if bench_args.profile:
        profiler.stop()
        profiler.export_chrome_trace(os.path.join(os.environ["SGLANG_TORCH_PROFILER_DIR"], f"debug_rank{tp_rank}.trace.json.gz"))
    

def generate_features(start, end, gpu_index, args):
    dataset = generate_data(args, args.dataset, start, end)
    print(f"GPU {gpu_index} dataset before all_gather: {dataset}")
    # torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    gathered_datasets = [None] * world_size
    torch.distributed.all_gather_object(gathered_datasets, dataset)
    combined_dataset = concatenate_datasets(gathered_datasets)
    print(f"GPU {gpu_index} dataset after all_gather: {combined_dataset}")
    bench_args = BenchArgs.from_cli_args(args)
    server_args = ServerArgs.from_cli_args(args)
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)
    server_args.cuda_graph_bs = list(bench_args.batch_size)
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    sgl_generate(combined_dataset, server_args, port_args, bench_args, gpu_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--max-token-length", type=int, default=2048)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sharegpt", "ultrachat", "mixture_of_thoughts"],
        default="sharegpt",
    )
    parser.add_argument(
        "--enable-fused-features",
        action="store_true",
        help="enable fused features for eagle3",
    )
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    assert world_size == args.tp_size, "support only one Engine for now"
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Saving to {args.outdir}", flush=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    s, e = 0, DATASET_INFO[args.dataset]["num_samples"]
    # s, e = 0, 128
    data_a = split_range(s, e, world_size, over=True)
    start, end = data_a[rank]
    generate_features(start, end, rank, args)

