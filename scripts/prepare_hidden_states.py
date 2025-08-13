"""
This script will generate the hidden states for the dataset.
By generating hidden states in advance, we can avoid:
- the memory overhead of loading target model
- the latency overhead of generating hidden states for each request.
"""

import argparse
import hashlib
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from sglang.bench_one_batch import BenchArgs, load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DeepEPMode,
    configure_logger,
    get_bool_env_var,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from specforge.data import build_eagle3_dataset
from specforge.utils import print_with_rank, rank_0_priority


class LogitsProcessorForEAGLE3(torch.nn.Module):
    def __init__(self, logits_processor: LogitsProcessor):
        super().__init__()
        self.logits_processor = logits_processor

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        ret = self.logits_processor.forward(
            input_ids, hidden_states, lm_head, logits_metadata, aux_hidden_states
        )
        ret.last_hidden_states = hidden_states
        return ret


def wrap_logits_processors_in_module(module: nn.Module):
    for name, submodule in module.named_modules():
        if isinstance(submodule, LogitsProcessor):
            wrapped = LogitsProcessorForEAGLE3(submodule)
            setattr(module, name, wrapped)
            print(f"wrapped {name} with LogitsProcessorForEAGLE3")


class SglangHiddenStatesGenerator:
    def __init__(self, args, tp_rank: int):
        self.args = args
        self.bench_args = BenchArgs.from_cli_args(args)
        self.server_args = ServerArgs.from_cli_args(args)
        self.server_args.enable_return_hidden_states = True
        self.server_args.context_length = args.max_length

        self.server_args.cuda_graph_max_bs = max(self.bench_args.batch_size)
        self.server_args.cuda_graph_bs = list(self.bench_args.batch_size)
        _set_envs_and_config(self.server_args)
        self.port_args = PortArgs.init_new(self.server_args)
        # Set CPU affinity
        if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
            set_gpu_proc_affinity(
                self.server_args.tp_size, self.server_args.nnodes, tp_rank
            )
        configure_logger(self.server_args, prefix=f" TP{tp_rank}")
        self.model_runner, _ = load_model(self.server_args, self.port_args, tp_rank)
        wrap_logits_processors_in_module(self.model_runner.model)
        self.tp_rank = tp_rank

        config = AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=self.server_args.trust_remote_code
        )
        if args.enable_aux_hidden_states and args.aux_hidden_states_layers is None:
            if hasattr(config, "num_hidden_layers"):
                num_layers = config.num_hidden_layers
            elif hasattr(config, "text_config"):
                num_layers = config.text_config.num_hidden_layers
            else:
                raise ValueError(
                    f"config {config} does not have num_hidden_layers or text_config.num_hidden_layers"
                )
            # in sglang, when we do set_eagle3_layers_to_capture, we will add 1 to the layer index
            args.aux_hidden_states_layers = [
                2 - 1,
                num_layers // 2 - 1,
                num_layers - 3 - 1,
            ]
            assert (
                len(args.aux_hidden_states_layers) == 3
            ), "aux_hidden_states_layers is expected to be 3 layers"
            print_with_rank(
                f"Capturing Aux hidden states layers: {args.aux_hidden_states_layers}, num_layers: {num_layers}"
            )

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch, model_runner):
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
    def _extend(self, reqs, model_runner, capture_aux_hidden_states):
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
        self._maybe_prepare_mlp_sync_batch(batch, model_runner)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, _ = model_runner.forward(forward_batch)
        aux_hidden_states_list = None
        input_lens = [len(req.origin_input_ids) for req in reqs]
        if capture_aux_hidden_states:
            assert (
                hasattr(logits_output, "last_hidden_states")
                and logits_output.last_hidden_states is not None
            ), "please use https://github.com/zyksir/sglang/tree/eagle3-offline"
            hidden_states_list = torch.split(
                logits_output.last_hidden_states, input_lens, dim=0
            )
            aux_hidden_states_list = torch.split(
                logits_output.hidden_states, input_lens, dim=0
            )
        else:
            hidden_states_list = torch.split(
                logits_output.hidden_states, input_lens, dim=0
            )
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        return hidden_states_list, aux_hidden_states_list

    def _save_tensor(self, hidden_states_cpu, save_aux_hidden_states):
        for idx, (hidden_states, batch_save_info) in enumerate(hidden_states_cpu):
            if idx % torch.distributed.get_world_size() != torch.distributed.get_rank():
                continue
            hidden_states_list, aux_hidden_states_list = hidden_states
            if save_aux_hidden_states:
                for hidden_state, aux_hidden_state, (data_point, output_file) in zip(
                    hidden_states_list, aux_hidden_states_list, batch_save_info
                ):
                    data_point["hidden_state"] = hidden_state.clone().unsqueeze(0).cpu()
                    data_point["aux_hidden_state"] = (
                        aux_hidden_state.clone().unsqueeze(0).cpu()
                    )
                    assert not torch.any(
                        torch.isnan(data_point["hidden_state"])
                    ), "hidden_state is expected to be non-nan"
                    assert not torch.any(
                        torch.isnan(data_point["aux_hidden_state"])
                    ), "aux_hidden_state is expected to be non-nan"
                    torch.save(data_point, output_file)
            else:
                for hidden_state, (data_point, output_file) in zip(
                    hidden_states_list, batch_save_info
                ):
                    data_point["hidden_state"] = hidden_state.clone().unsqueeze(0).cpu()
                    assert not torch.any(
                        torch.isnan(data_point["hidden_state"])
                    ), "hidden_state is expected to be non-nan"
                    torch.save(data_point, output_file)

    def generate(self, dataset: Dataset):
        MIN_FILE_SIZE = 100 * 1024
        if self.args.enable_aux_hidden_states:
            if not hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
                raise ValueError(
                    f"model_runner.model {self.model_runner.model} does not have set_eagle3_layers_to_capture"
                )
            self.model_runner.model.set_eagle3_layers_to_capture(
                self.args.aux_hidden_states_layers
            )
            if hasattr(self.model_runner.model, "capture_aux_hidden_states"):
                assert (
                    self.model_runner.model.capture_aux_hidden_states
                ), "model_runner.model.capture_aux_hidden_states is expected to be True"
            elif hasattr(
                self.model_runner.model.language_model, "capture_aux_hidden_states"
            ):
                assert (
                    self.model_runner.model.language_model.capture_aux_hidden_states
                ), "model_runner.model.capture_aux_hidden_states is expected to be True"
            else:
                raise ValueError(
                    f"model_runner.model {self.model_runner.model} does not have capture_aux_hidden_states"
                )

        if self.bench_args.profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )
            profiler.start()
        # Prepare inputs for warm up
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs = []
        hidden_states_cpu = []
        batch_size = self.bench_args.batch_size[0]
        batch_save_info = []
        group_size = 5000
        for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
            group_start = (idx // group_size) * group_size
            group_end = group_start + group_size
            grouped_subdir = f"rows_{group_start}-{group_end}"
            if self.tp_rank == 0 and not os.path.exists(
                f"{self.args.output_path}/{grouped_subdir}"
            ):
                os.makedirs(f"{self.args.output_path}/{grouped_subdir}")

            output_file = f"{self.args.output_path}/{grouped_subdir}/data_{idx}.ckpt"
            if (
                os.path.exists(output_file)
                and os.path.getsize(output_file) > MIN_FILE_SIZE
            ):
                continue

            batch_save_info.append(
                (
                    {
                        "input_ids": row["input_ids"].view(-1),
                        "loss_mask": row["loss_mask"].view(-1),
                    },
                    output_file,
                )
            )

            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=row["input_ids"].view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)
            if len(reqs) == batch_size:
                hidden_states_list = self._extend(
                    reqs, self.model_runner, self.args.enable_aux_hidden_states
                )
                hidden_states_cpu.append((hidden_states_list, batch_save_info[:]))
                if len(hidden_states_cpu) >= 64:
                    torch.cuda.synchronize()
                    self._save_tensor(
                        hidden_states_cpu,
                        save_aux_hidden_states=self.args.enable_aux_hidden_states,
                    )
                    hidden_states_cpu = []
                    torch.cuda.empty_cache()
                batch_save_info, reqs = [], []

        torch.cuda.synchronize()
        self._save_tensor(
            hidden_states_cpu, save_aux_hidden_states=self.args.enable_aux_hidden_states
        )
        if self.bench_args.profile:
            profiler.stop()
            profiler.export_chrome_trace(
                os.path.join(
                    os.environ["SGLANG_TORCH_PROFILER_DIR"],
                    f"debug_rank{self.tp_rank}.trace.json.gz",
                )
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    # parser.add_argument("--chat-template", type=str, default="llama3")

    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--enable-aux-hidden-states", action="store_true")
    parser.add_argument("--aux-hidden-states-layers", type=str, default=None)
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # args.dist_timeout is defined in sglang.srt.server_args.ServerArgs and is in seconds.
    if args.dist_timeout is not None:
        if args.dist_timeout <= 0:
            raise ValueError(
                f"--dist-timeout must be a positive number of seconds, but got {args.dist_timeout}"
            )
        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(seconds=args.dist_timeout)
        )
    else:
        torch.distributed.init_process_group(backend="nccl")

    assert os.path.exists(
        args.data_path
    ), f"Dataset path {args.data_path} does not exist"
    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        args.output_path = root_path.joinpath("cache", "hidden_states")

    dataset = load_dataset("json", data_files=args.data_path)["train"]
    if args.num_samples is not None:
        dataset = dataset.select(range(args.num_samples))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    cache_params_string = (
        f"{args.data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.model_path}"  # Tokenizer may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    with rank_0_priority():
        eagle3_dataset = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
        )
        print_with_rank("Built dataset")

    hidden_states_generator = SglangHiddenStatesGenerator(
        args, tp_rank=torch.distributed.get_rank()
    )
    hidden_states_generator.generate(eagle3_dataset)


if __name__ == "__main__":
    main()
