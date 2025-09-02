"""
This file contains the wrapper for the SGL model.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from sglang.bench_one_batch import BenchArgs, _maybe_prepare_mlp_sync_batch, load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import configure_logger, get_bool_env_var, set_gpu_proc_affinity
from transformers import AutoConfig

from specforge.data.preprocessing import OfflineEagle3Dataset
from specforge.utils import print_with_rank


class LogitsProcessorForEAGLE3(torch.nn.Module):
    def __init__(
        self, logits_processor: LogitsProcessor, return_full_logits: bool = False
    ):
        super().__init__()
        self.logits_processor = logits_processor
        self.return_full_logits = return_full_logits

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        if self.return_full_logits:
            logits_metadata.forward_mode = ForwardMode.DECODE
        ret = self.logits_processor.forward(
            input_ids, hidden_states, lm_head, logits_metadata, aux_hidden_states
        )
        if self.return_full_logits:
            ret.last_hidden_states = ret.next_token_logits
        else:
            ret.last_hidden_states = hidden_states
        return ret


def wrap_logits_processors_in_module(
    module: nn.Module, return_full_logits: bool = False
):
    for name, submodule in module.named_modules():
        if isinstance(submodule, LogitsProcessor):
            wrapped = LogitsProcessorForEAGLE3(submodule, return_full_logits)
            setattr(module, name, wrapped)
            print(f"wrapped {name} with LogitsProcessorForEAGLE3")


@torch.no_grad
def _extend(
    reqs, model_runner, capture_aux_hidden_states: bool, return_full_logits: bool
):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
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
        hidden_states_list = torch.split(logits_output.hidden_states, input_lens, dim=0)
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    return hidden_states_list, aux_hidden_states_list


class SglangTargetModel(nn.Module):
    def __init__(
        self,
        args,
        target_micro_batch_size,
        draft_micro_batch_size,
        tp_group,
        enable_aux_hidden_states=True,
        return_full_logits=False,
    ):
        super().__init__()
        self.return_full_logits = return_full_logits
        tp_rank = dist.get_rank(group=tp_group)
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.target_tp_size = args.target_tp_size
        self.target_micro_batch_size = target_micro_batch_size
        self.draft_micro_batch_size = draft_micro_batch_size
        assert draft_micro_batch_size == 1, "draft_micro_batch_size must be 1 for now"
        self.enable_aux_hidden_states = enable_aux_hidden_states
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
        tp_rank = dist.get_rank(group=tp_group)
        self.model_runner, _ = load_model(self.server_args, self.port_args, tp_rank)
        wrap_logits_processors_in_module(self.model_runner.model, return_full_logits)

    def set_aux_hidden_states_layers(self, aux_hidden_states_layers=None):
        config = AutoConfig.from_pretrained(
            self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
        )
        if not self.enable_aux_hidden_states:
            return
        if aux_hidden_states_layers is None:
            if hasattr(config, "num_hidden_layers"):
                num_layers = config.num_hidden_layers
            elif hasattr(config, "text_config"):
                num_layers = config.text_config.num_hidden_layers
            else:
                raise ValueError(
                    f"config {config} does not have num_hidden_layers or text_config.num_hidden_layers"
                )
            # in sglang, when we do set_eagle3_layers_to_capture, we will add 1 to the layer index
            aux_hidden_states_layers = [
                2 - 1,
                num_layers // 2 - 1,
                num_layers - 3 - 1,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers"
        print_with_rank(
            f"Capturing Aux hidden states layers: {self.aux_hidden_states_layers}"
        )

        if not hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            raise ValueError(
                f"model_runner.model {self.model_runner.model} does not have set_eagle3_layers_to_capture"
            )
        self.model_runner.model.set_eagle3_layers_to_capture(
            self.aux_hidden_states_layers
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

    def extend(self, reqs: List[Req]) -> List[Dict[str, torch.Tensor]]:
        return _extend(
            reqs,
            self.model_runner,
            self.enable_aux_hidden_states,
            self.return_full_logits,
        )

    def forward(
        self,
        data_for_target: List[Dict[str, torch.Tensor]],
        draft_data_collator,
        draft_dp_rank: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        arguments:
            data_for_target: List[Dict[str, torch.Tensor]] of target_batch_size
                - input_ids: (tp_size, seq_len)
                - attention_mask: (tp_size, seq_len)
                - loss_mask: (tp_size, seq_len)
        return:
            data_for_draft: List[Dict[str, torch.Tensor]] of draft_batch_size, draft_micro_batch_size = 1
                - input_ids: (1, seq_len)
                - attention_mask: (1, seq_len)
                - loss_mask: (1, seq_len)
                - target: (1, seq_len, vocab_size) or (1, seq_len, hidden_size)
                - hidden_states: (1, seq_len, hidden_size)
        """
        num_items = len(data_for_target)
        target_total = (
            math.ceil(num_items / self.target_micro_batch_size)
            * self.target_micro_batch_size
        )
        padding_needed = target_total - num_items
        data_for_target = data_for_target + data_for_target[:padding_needed]

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs, data_cache = [], []
        data_for_draft = []
        for idx_data, data in enumerate(data_for_target):
            assert (
                data["input_ids"].shape[0] == self.target_tp_size
            ), "input_ids.shape[0] must be equal to target_tp_size"
            for idx_row in range(data["input_ids"].shape[0]):
                req = Req(
                    rid=str(idx_row + idx_data * self.target_tp_size),
                    origin_input_text="",
                    origin_input_ids=data["input_ids"][idx_row].view(-1).tolist(),
                    sampling_params=sampling_params,
                )
                req.prefix_indices = []
                req.fill_ids = req.origin_input_ids
                req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
                req.logprob_start_len = len(req.origin_input_ids) - 1
                data_cache.append(data)
                reqs.append(req)
                if len(reqs) == self.target_micro_batch_size:
                    # here let me assume return aux_hidden_states is True
                    hidden_states_list, aux_hidden_states_list = self.extend(reqs)
                    for idx, (data, hidden_states, aux_hidden_states) in enumerate(
                        zip(data_cache, hidden_states_list, aux_hidden_states_list)
                    ):
                        if idx % dist.get_world_size() != draft_dp_rank:
                            continue
                        # the input shape is aligned with "prepare_hidden_states.py"
                        # the output shape is aligned with OfflineEagle3Dataset
                        data_for_draft.append(
                            OfflineEagle3Dataset.process_data(
                                {
                                    "input_ids": data["input_ids"].view(-1),
                                    "loss_mask": data["loss_mask"].view(-1),
                                    "hidden_state": hidden_states.unsqueeze(0),
                                    "aux_hidden_state": aux_hidden_states.unsqueeze(0),
                                },
                                transform=None,
                                max_len=self.args.max_length,
                            )
                        )
                    reqs, data_cache = [], []
        # for now, let us assume draft_micro_batch_size = 1
        return [draft_data_collator([data]) for data in data_for_draft]
