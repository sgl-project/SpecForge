import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

# --- Assuming these imports are correctly located in your project ---
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from specforge.data.preprocessing import OfflineEagle3Dataset
from specforge.distributed import get_tp_device_mesh, get_tp_group  # Assumed utility
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


class Eagle3TargetModel(ABC, nn.Module):
    """
    Abstract base class for target models in Eagle3.

    This class encapsulates common batching logic and defines a clear interface
    for different model-loading backends (e.g., Transformers, SGLang).
    Its primary role is to compute hidden states for given input sequences.
    """

    def __init__(
        self,
        model_name_or_path: str,
        target_micro_batch_size: int,
        draft_micro_batch_size: int = 1,
        enable_aux_hidden_states: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.target_micro_batch_size = target_micro_batch_size
        self.draft_micro_batch_size = draft_micro_batch_size
        self.enable_aux_hidden_states = enable_aux_hidden_states
        self.model_name_or_path = model_name_or_path
        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.hidden_size = self.config.hidden_size
        self.aux_hidden_states_layers = None

    @abstractmethod
    def _load_model(self) -> None:
        """Subclasses must implement this method to load the model and tokenizer."""
        raise NotImplementedError

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """
        Sets which intermediate layers to capture for auxiliary hidden states.

        Args:
            aux_hidden_states_layers: A list of layer indices (0-indexed) to capture.
                                      If None, a default set of layers is used.
        """
        if not self.enable_aux_hidden_states:
            self.aux_hidden_states_layers = []
            print("Auxiliary hidden states are disabled.")
            return

        if aux_hidden_states_layers is None:
            if hasattr(self.config, "num_hidden_layers"):
                num_layers = self.config.num_hidden_layers
            elif hasattr(self.config, "text_config"):
                num_layers = self.config.text_config.num_hidden_layers
            else:
                raise ValueError(
                    f"config {self.config} does not have num_hidden_layers or text_config.num_hidden_layers"
                )
            # in sglang, when we do set_eagle3_layers_to_capture, we will add 1 to the layer index
            aux_hidden_states_layers = [
                2 - 1,
                num_layers // 2 - 1,
                num_layers - 3 - 1,
            ]

        # A check can be placed here if a specific number of layers is required by the draft model.
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers"
        print_with_rank(
            f"Capturing Aux hidden states layers: {self.aux_hidden_states_layers}"
        )

    @abstractmethod
    @torch.no_grad()
    def _get_hidden_states_for_batch(
        self, reqs: List[Req]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        The core implementation for a single forward pass on a batch of requests.

        Args:
            reqs: A list of Req objects representing the inputs for the micro-batch.

        Returns:
            A tuple containing:
            - hidden_states_list: A list of final-layer hidden states [seq_len, hidden_size].
            - aux_hidden_states_list: A list of concatenated auxiliary hidden states
                                      [seq_len, N * hidden_size], or None if disabled.
        """
        raise NotImplementedError

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
        target_total = (  # target model forward times
            math.ceil(num_items / self.target_micro_batch_size)
            * self.target_micro_batch_size
        )
        padding_needed = target_total - num_items
        data_for_target = data_for_target + data_for_target[:padding_needed]

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs, data_cache = [], []
        data_for_draft = []
        for idx, data in enumerate(data_for_target):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=data["input_ids"].view(-1).tolist(),
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
                with torch.no_grad():
                    hidden_states_list, aux_hidden_states_list = (
                        self._get_hidden_states_for_batch(reqs)
                    )
                    for idx, (data, hidden_states, aux_hidden_states) in enumerate(
                        zip(data_cache, hidden_states_list, aux_hidden_states_list)
                    ):
                        if idx % torch.distributed.get_world_size() != draft_dp_rank:
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
                                max_len=self.max_length,
                            )
                        )
                    reqs, data_cache = [], []
        return [draft_data_collator([data]) for data in data_for_draft]


class TransformersTargetModel(Eagle3TargetModel):
    """
    Target model implementation using Hugging Face Transformers.
    Supports both standard `AutoModelForCausalLM` and custom distributed loaders.
    """

    def __init__(
        self,
        args,
        target_micro_batch_size: int,
        draft_micro_batch_size: int = 1,
        enable_aux_hidden_states: bool = True,
        trust_remote_code: bool = True,
        loader_type: str = "hf",
    ):
        super().__init__(
            args.target_model_path,
            target_micro_batch_size,
            draft_micro_batch_size,
            enable_aux_hidden_states,
            trust_remote_code,
        )
        self.max_length = args.max_length
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        self._load_model(
            loader_type=loader_type,
            torch_dtype=self.config.torch_dtype if self.config.torch_dtype else "auto",
        )

    def _load_model(self, loader_type: str, torch_dtype: Any) -> None:
        print_with_rank(
            f"Loading model with Transformers backend (loader: {loader_type})..."
        )
        if loader_type == "hf":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                tp_plan="auto",
                device_mesh=get_tp_device_mesh(),
            )
        elif loader_type == "custom":
            from specforge.modeling.auto import AutoDistributedTargetModel

            self.model = AutoDistributedTargetModel.from_pretrained(
                self.model_name_or_path,
                device=self.device,
                torch_dtype=self.config.torch_dtype or torch_dtype,
            )
        else:
            raise ValueError(
                f"Unknown loader_type: '{loader_type}'. Choose 'hf' or 'custom'."
            )
        if loader_type == "custom":
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def _get_hidden_states_for_batch(
        self, reqs: List[Req]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        if not reqs:
            return [], None if self.enable_aux_hidden_states else []

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

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id or 0,
        )
        attention_mask = (padded_input_ids != (self.tokenizer.pad_token_id or 0)).long()
        outputs = self.model(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        all_hidden_states = outputs.hidden_states
        last_layer_hs = outputs.hidden_states[-1]  # [B, L, H]

        aux_hidden_states_list = None
        if self.enable_aux_hidden_states:
            assert self.aux_hidden_states_layers is not None, "Aux layers not set"
            aux_hidden_states_list = []
            for i, seq_len in enumerate(input_lens_sorted):
                layer_tensors = []
                for layer_idx in self.aux_hidden_states_layers:
                    hs = all_hidden_states[layer_idx + 1][i, :seq_len, :].detach()
                    layer_tensors.append(hs)
                aux_concat = torch.cat(layer_tensors, dim=-1)  # [seq_len, 3*H]
                aux_hidden_states_list.append(aux_concat)

        hidden_states_list_sorted = [
            last_layer_hs[i, :seq_len, :].detach()
            for i, seq_len in enumerate(input_lens_sorted)
        ]

        inv_idx = [sorted_idx.index(i) for i in range(len(sorted_idx))]
        hidden_states_list = [hidden_states_list_sorted[i] for i in inv_idx]
        aux_hidden_states_list = (
            [aux_hidden_states_list[i] for i in inv_idx]
            if aux_hidden_states_list
            else None
        )
        return hidden_states_list, aux_hidden_states_list


class SGLangTargetModel(Eagle3TargetModel):
    """Target model implementation using the SGLang backend."""

    def __init__(
        self,
        args,
        target_micro_batch_size: int,
        draft_micro_batch_size: int = 1,
        enable_aux_hidden_states: bool = True,
        return_full_logits=False,
    ):
        super().__init__(
            args.target_model_path,
            target_micro_batch_size,
            draft_micro_batch_size,
            enable_aux_hidden_states,
            True,
        )
        self.max_length = args.max_length
        self.return_full_logits = return_full_logits
        self.tp_rank = dist.get_rank(group=get_tp_group())
        self.target_tp_size = args.target_tp_size
        self._load_model(args)

    def _load_model(self, args) -> None:
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
                self.server_args.tp_size, self.server_args.nnodes, self.tp_rank
            )
        configure_logger(self.server_args, prefix=f" TP{self.tp_rank}")
        self.model_runner, _ = load_model(
            self.server_args, self.port_args, self.tp_rank
        )
        wrap_logits_processors_in_module(
            self.model_runner.model, self.return_full_logits
        )

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """Extends base method to configure SGLang's specific layer capture mechanism."""
        super().set_aux_hidden_states_layers(aux_hidden_states_layers)
        if self.enable_aux_hidden_states and self.aux_hidden_states_layers:
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

    @torch.no_grad()
    def _get_hidden_states_for_batch(
        self, reqs: List[Req]
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        if not reqs:
            return [], None

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=None,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        _maybe_prepare_mlp_sync_batch(batch, self.model_runner)
        forward_batch = ForwardBatch.init_new(
            batch.get_model_worker_batch(), self.model_runner
        )
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, _ = self.model_runner.forward(forward_batch)
        input_lens = [len(req.origin_input_ids) for req in reqs]

        if self.enable_aux_hidden_states:
            assert (
                hasattr(logits_output, "last_hidden_states")
                and logits_output.last_hidden_states is not None
            ), "the sglang version is outdated, please upgrade sglang"
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
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        return hidden_states_list, aux_hidden_states_list


class TargetModelFactory:
    """A factory class is used to create and return instances of the target model based on the configuration."""

    @staticmethod
    def create(
        args,
        target_micro_batch_size: int,
        draft_micro_batch_size: int,
        enable_aux_hidden_states: bool = True,
        return_full_logits: bool = False,
    ) -> Eagle3TargetModel:
        backend = args.target_model_backend.lower()

        if backend in ["hf", "custom"]:
            return TransformersTargetModel(
                args=args,
                target_micro_batch_size=target_micro_batch_size,
                draft_micro_batch_size=draft_micro_batch_size,
                enable_aux_hidden_states=enable_aux_hidden_states,
                trust_remote_code=True,
                loader_type=backend,
            )
        elif backend == "sglang":
            return SGLangTargetModel(
                args=args,
                target_micro_batch_size=target_micro_batch_size,
                draft_micro_batch_size=draft_micro_batch_size,
                enable_aux_hidden_states=enable_aux_hidden_states,
                return_full_logits=return_full_logits,
            )
        else:
            raise ValueError(
                f"Unknown target model backend: '{args.target_model_backend}'. Valid options are 'hf', 'custom', 'sglang'."
            )
