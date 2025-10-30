"""
This script will generate the hidden states for the dataset use transformer as the target model backend.
By generating hidden states in advance, we can avoid:
- the memory overhead of loading target model
- the latency overhead of generating hidden states for each request.
"""

import argparse
import hashlib
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset, load_dataset
from sglang.bench_one_batch import BenchArgs, load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from specforge.data import build_eagle3_dataset
from specforge.distributed import (
    destroy_distributed,
    get_device_mesh,
    get_dp_group,
    get_tp_group,
    init_distributed,
    is_tp_rank_0,
    print_rank_info,
)
from specforge.modeling.auto import AutoDistributedTargetModel
from specforge.modeling.target.target_model import TargetModelFactory
from specforge.utils import print_with_rank


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
    def __init__(
        self,
        args,
        global_rank: int,
        tp_rank: int,
        dp_rank: int,
        dp_group,
        tp_size: int,
        dp_size: int,
    ):
        self.args = args
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.dp_group = dp_group
        self.tp_size = tp_size
        self.dp_size = dp_size

        self.model_name_or_path = args.model_path
        self.bench_args = BenchArgs.from_cli_args(args)
        self.server_args = ServerArgs.from_cli_args(args)
        self.server_args.enable_return_hidden_states = True
        self.server_args.context_length = args.max_length
        self.server_args.cuda_graph_max_bs = max(self.bench_args.batch_size)
        self.server_args.cuda_graph_bs = list(self.bench_args.batch_size)
        _set_envs_and_config(self.server_args)

        self.port_args = PortArgs.init_new(self.server_args)
        if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
            set_gpu_proc_affinity(
                self.server_args.tp_size, self.server_args.nnodes, tp_rank
            )
        configure_logger(self.server_args, prefix=f" DP{dp_rank} TP{tp_rank}")

        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        print_rank_info()
        args.target_model_backend = "hf"
        self.model = TargetModelFactory.create(
            args=self.args,
            target_micro_batch_size=args.tp_size,
            draft_micro_batch_size=1,
            enable_aux_hidden_states=True,
            return_full_logits=False,
        ).model
        self.model.eval()

        # Set aux layers
        self.enable_aux_hidden_states = args.enable_aux_hidden_states
        if self.enable_aux_hidden_states:
            self.set_aux_hidden_states_layers(args.aux_hidden_states_layers)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ):
        if not self.enable_aux_hidden_states:
            self.aux_hidden_states_layers = []
            return

        if aux_hidden_states_layers is None:
            num_layers = getattr(self.config, "num_hidden_layers", None)
            if num_layers is None and hasattr(self.config, "text_config"):
                num_layers = self.config.text_config.num_hidden_layers
            if num_layers is None:
                raise ValueError("Cannot infer number of layers from config.")
            # Default: 3 layers as in SGLang implementation
            aux_hidden_states_layers = [
                2 - 1,
                num_layers // 2 - 1,
                num_layers - 3 - 1,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "Expected exactly 3 auxiliary layers."

    @torch.no_grad()
    def extend(
        self, reqs: List[Req]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        if not reqs:
            return [], None

        input_ids_list = [
            torch.tensor(
                req.origin_input_ids, dtype=torch.long, device=self.model.device
            )
            for req in reqs
        ]
        input_lens = [len(ids) for ids in input_ids_list]

        sorted_idx = sorted(
            range(len(input_ids_list)), key=lambda i: input_lens[i], reverse=True
        )
        input_ids_list = [input_ids_list[i] for i in sorted_idx]
        input_lens_sorted = [input_lens[i] for i in sorted_idx]

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        attention_mask = (padded_input_ids != pad_id).long()

        outputs = self.model(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_layer_hs = outputs.hidden_states[-1]  # [B, L, H]

        # Extract only specified aux layers
        aux_hidden_states_list_sorted = None
        if self.enable_aux_hidden_states:
            aux_hidden_states_list_sorted = []
            for i, seq_len in enumerate(input_lens_sorted):
                selected = []
                for layer_idx in self.aux_hidden_states_layers:
                    hs = outputs.hidden_states[layer_idx + 1][i, :seq_len, :].detach()
                    selected.append(hs)
                aux_concat = torch.cat(selected, dim=-1)  # [seq_len, 3*H]
                aux_hidden_states_list_sorted.append(aux_concat)

        hidden_states_list_sorted = [
            last_layer_hs[i, :seq_len, :].detach()
            for i, seq_len in enumerate(input_lens_sorted)
        ]

        # more efficient than above
        sorted_idx_tensor = torch.tensor(sorted_idx)
        inv_idx = torch.argsort(sorted_idx_tensor).tolist()

        hidden_states_list = [hidden_states_list_sorted[i] for i in inv_idx]
        aux_hidden_states_list = (
            [aux_hidden_states_list_sorted[i] for i in inv_idx]
            if aux_hidden_states_list_sorted
            else None
        )

        return hidden_states_list, aux_hidden_states_list

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch, model_runner):
        if require_mlp_sync(model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                enable_two_batch_overlap=model_runner.server_args.enable_two_batch_overlap,
                enable_deepep_moe=MoeA2ABackend(
                    model_runner.server_args.moe_a2a_backend
                ).is_deepep(),
                deepep_mode=DeepEPMode(model_runner.server_args.deepep_mode),
                require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
                disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            )

    def _save_tensor(self, hidden_states_cpu, save_aux_hidden_states):
        assert self.tp_rank == 0, "Only tp_rank=0 should call _save_tensor"
        for (
            hidden_states_list,
            aux_hidden_states_list,
        ), batch_save_info in hidden_states_cpu:
            if save_aux_hidden_states:
                for hs, aux_hs, (data_point, output_file) in zip(
                    hidden_states_list, aux_hidden_states_list, batch_save_info
                ):
                    data_point["hidden_state"] = hs.clone().unsqueeze(0).cpu()
                    data_point["aux_hidden_state"] = aux_hs.clone().unsqueeze(0).cpu()
                    assert not torch.any(
                        torch.isnan(data_point["hidden_state"])
                    ), "NaN in hidden_state"
                    assert not torch.any(
                        torch.isnan(data_point["aux_hidden_state"])
                    ), "NaN in aux_hidden_state"
                    torch.save(data_point, output_file)
            else:
                for hs, (data_point, output_file) in zip(
                    hidden_states_list, batch_save_info
                ):
                    data_point["hidden_state"] = hs.clone().unsqueeze(0).cpu()
                    assert not torch.any(
                        torch.isnan(data_point["hidden_state"])
                    ), "NaN in hidden_state"
                    torch.save(data_point, output_file)

    def generate(self, dataset: Dataset, start_index: int = 0):
        MIN_FILE_SIZE = 100 * 1024
        if self.args.enable_aux_hidden_states:
            self.set_aux_hidden_states_layers(self.args.aux_hidden_states_layers)

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs = []
        hidden_states_cpu = []
        batch_size = self.bench_args.batch_size[0]
        batch_save_info = []
        group_size = 2000

        for local_idx, row in tqdm(
            enumerate(dataset),
            total=len(dataset),
            disable=(self.tp_rank != 0),
            desc=f"Global Rank:{self.global_rank} Processing",
        ):
            global_idx = start_index + local_idx
            group_idx = (local_idx // group_size) * group_size
            group_start = start_index + group_idx
            group_end = group_start + min(group_size, len(dataset) - group_idx)
            grouped_subdir = f"rows_{group_start}-{group_end}"

            output_file = (
                f"{self.args.output_path}/{grouped_subdir}/data_{global_idx}.ckpt"
            )

            if self.tp_rank == 0:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
                rid=str(global_idx),
                origin_input_text="",
                origin_input_ids=row["input_ids"].view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)

            if len(reqs) == batch_size:
                hidden_states_list, aux_hidden_states_list = self.extend(reqs)
                hidden_states_cpu.append(
                    ((hidden_states_list, aux_hidden_states_list), batch_save_info[:])
                )

                batch_save_info, reqs = [], []

                if len(hidden_states_cpu) >= 64:
                    torch.cuda.synchronize()
                    if self.tp_rank == 0:
                        self._save_tensor(
                            hidden_states_cpu, self.args.enable_aux_hidden_states
                        )
                    hidden_states_cpu = []
                    torch.cuda.empty_cache()

        # handle last batch which is not full
        if reqs:
            hidden_states_list, aux_hidden_states_list = self.extend(reqs)
            hidden_states_cpu.append(
                ((hidden_states_list, aux_hidden_states_list), batch_save_info[:])
            )

        torch.cuda.synchronize()
        if self.tp_rank == 0 and hidden_states_cpu:
            self._save_tensor(hidden_states_cpu, self.args.enable_aux_hidden_states)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chat-template", type=str, default="llama3")

    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--enable-aux-hidden-states", action="store_true")
    parser.add_argument("--aux-hidden-states-layers", type=str, default=None)
    parser.add_argument("--build-dataset-num-proc", type=int, default=8)

    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    init_distributed(
        timeout=args.dist_timeout or 300000, tp_size=args.tensor_parallel_size
    )

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    dp_rank, tp_rank = get_device_mesh().get_coordinate()
    dp_size = world_size // args.tensor_parallel_size
    tp_size = args.tensor_parallel_size

    dp_group = get_dp_group()
    tp_group = get_tp_group()

    print_with_rank(
        f"Global Rank {global_rank}: "
        f"DP Rank {dp_rank}, TP Rank {tp_rank}, DP Size {dp_size}, TP Size {tp_size}"
    )

    if args.aux_hidden_states_layers is not None:
        args.aux_hidden_states_layers = [
            int(x) for x in args.aux_hidden_states_layers.split(",")
        ]

    assert os.path.exists(
        args.data_path
    ), f"Dataset path {args.data_path} does not exist"
    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        args.output_path = root_path / "cache" / "hidden_states"

    dataset = load_dataset("json", data_files=args.data_path)["train"]
    if args.num_samples is not None:
        dataset = dataset.select(range(args.num_samples))

    total = len(dataset)
    per_dp = total // dp_size
    start = dp_rank * per_dp
    end = start + per_dp if dp_rank < dp_size - 1 else total
    dataset_shard = dataset.select(range(start, end))
    print(
        f"DP Rank {dp_rank}: processing \033[91m{len(dataset_shard)}\033[0m samples (\033[91m{start}\033[0m-\033[91m{end}\033[0m)"
    )

    # Tokenizer 和cache
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    cache_params_string = f"{args.data_path}-{args.max_length}-{args.chat_template}-{args.model_path}-{args.num_samples}-{start}-{end}"
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    # TP Rank 0 construct its dataset cache
    if is_tp_rank_0():
        print_with_rank("TP rank 0: Building dataset cache...")
        eagle3_shard = build_eagle3_dataset(
            dataset=dataset_shard,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            num_proc=args.build_dataset_num_proc,
        )
    else:
        print_with_rank("Not TP rank 0: Skipping dataset building.")

    # === All TP rank synchronization: Ensure cache has been written. ===
    dist.barrier(group=get_tp_group())
    # === All ranks (including non-TP-0) load cached data. ===
    print_with_rank("Loading cached dataset...")
    if not is_tp_rank_0():
        eagle3_shard = build_eagle3_dataset(
            dataset=dataset_shard,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            num_proc=1,
        )
    hidden_states_generator = SglangHiddenStatesGenerator(
        args,
        global_rank=global_rank,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        dp_group=dp_group,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    hidden_states_generator.generate(eagle3_shard, start_index=start)
    destroy_distributed()


if __name__ == "__main__":
    main()
