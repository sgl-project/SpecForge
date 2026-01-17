from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sglang.srt.managers.mm_utils as mm_utils
import torch
import torch.distributed as dist
import torch.nn as nn
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_device_mesh, get_tp_group
from specforge.utils import padding

from .sglang_backend import SGLangRunner, wrap_eagle3_logits_processors_in_module
from .sglang_backend.utils import LogitsProcessorForEAGLE3


@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_states: Optional[torch.Tensor] = None


class Eagle3TargetModel(ABC):
    """
    This  offers a layer of abstraction for the target model backend. The user can choose different backends to suit their needs:
    1. SGLang backend: for the mainstream model support with the fastest inference speed
    2. HuggingFace backend: for models that are not supported by SGLang but can be loaded by HuggingFace.
    3. Custom backend: for models with customized architecture and inference plan.
    """

    def __init__(self):
        self.aux_hidden_states_layers = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Eagle3TargetModel":
        """
        Initialize the target model backend from a pretrained model path.
        """

    @abstractmethod
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        """
        Generate the eagle3 data from the target model.
        """

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """
        Set the layers to capture the aux hidden states from the target model outputs.
        """
        if aux_hidden_states_layers is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
            else:
                raise ValueError(
                    f"Failed to set aux hidden states layers as model config {self.model.config} does not have num_hidden_layers"
                )
            aux_hidden_states_layers = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers for EAGLE3"


class HFEagle3TargetModel(Eagle3TargetModel):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "HFEagle3TargetModel":
        """
        Initialize the HuggingFace target model backend from a pretrained model path.
        """
        tp_size = get_tp_group().size()

        if tp_size > 1:
            device_kwargs = {
                "tp_plan": "auto",
                "tp_size": tp_size,
                "device_mesh": get_tp_device_mesh(),
            }
        else:
            device_kwargs = {
                "device_map": device,
            }

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **device_kwargs,
            **kwargs,
        )
        return cls(target_model)

    def _get_transformer_layers(self):
        """
        Helper to find the module list containing the transformer layers.
        Adapts to common architectures (Llama, Qwen, Mistral, OPT, etc.)
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        else:
            raise ValueError(
                "Could not locate transformer layers in the model architecture to register hooks."
            )

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        """
        Optimized HF backend:
        Instead of returning all hidden states (memory heavy), we use forward hooks
        to capture only the specific layers required by Eagle3.
        """
        captured_states = {}
        handles = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                # HF outputs for layers are usually tuples (hidden_states, present_key_value, ...)
                # We only need the hidden_states (first element)
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured_states[layer_idx] = hidden

            return hook

        # Locate the transformer layers ModuleList
        layers = self._get_transformer_layers()

        target_indices = self.aux_hidden_states_layers

        # Register hooks
        for idx in target_indices:
            # Ensure index is within bounds
            if 0 <= idx < len(layers):
                handles.append(layers[idx].register_forward_hook(get_hook(idx)))
            else:
                raise ValueError(
                    f"Layer index {idx} out of bounds for model with {len(layers)} layers."
                )

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                output_router_logits=False,
                use_cache=False,
            )
            target = outputs.logits
        finally:
            # Always remove hooks to prevent memory leaks or side effects on subsequent calls
            for handle in handles:
                handle.remove()

        # Verify we captured everything
        if len(captured_states) != 3:
            raise RuntimeError(
                f"Expected to capture 3 layers, but captured {len(captured_states)}"
            )

        # Extract in the correct order
        hidden_states0 = captured_states[target_indices[0]]
        hidden_states1 = captured_states[target_indices[1]]
        hidden_states2 = captured_states[target_indices[2]]

        hidden_states = torch.cat(
            (hidden_states0, hidden_states1, hidden_states2), dim=-1
        )

        # apply pading
        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(target.device)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class SGLangEagle3TargetModel(Eagle3TargetModel):

    def __init__(self, model_runner: SGLangRunner, hf_config=None):
        super().__init__()
        self.model_runner = model_runner
        self.hf_config = hf_config

        # VLM-specific attributes (initialized from hf_config if available)
        self._init_vlm_attributes()

    def _init_vlm_attributes(self):
        """Initialize VLM-specific attributes from hf_config for models like Qwen2.5-VL"""
        if self.hf_config is None:
            self.is_vlm = False
            return

        # Check if this is a VLM model by looking for vision_config
        self.is_vlm = hasattr(self.hf_config, "vision_config")

        if not self.is_vlm:
            return

        init_mm_embedding_cache(1024 * 1024 * 512)
        # Model type (e.g., "qwen2_5_vl", "qwen2_vl")
        self.model_type = getattr(self.hf_config, "model_type", None)

        # Vision config attributes
        vision_config = self.hf_config.vision_config
        self.spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        self.tokens_per_second = getattr(vision_config, "tokens_per_second", None)

        # Special token IDs from hf_config
        self.image_token_id = getattr(self.hf_config, "image_token_id", None)
        self.video_token_id = getattr(self.hf_config, "video_token_id", None)
        self.vision_start_token_id = getattr(
            self.hf_config, "vision_start_token_id", None
        )
        self.vision_end_token_id = getattr(self.hf_config, "vision_end_token_id", None)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangEagle3TargetModel":
        tp_size = dist.get_world_size(get_tp_group())
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            enable_return_hidden_states=True,
            disable_cuda_graph=True,  # we use piecewise cuda graph for prefill instead
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=dist.get_rank(get_tp_group()),
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        wrap_eagle3_logits_processors_in_module(
            model_runner.model, return_full_logits=False
        )

        # Get hf_config from model_config for VLM attributes
        hf_config = getattr(model_config, "hf_config", None)

        return cls(model_runner, hf_config=hf_config)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        self.model_runner.model.set_eagle3_layers_to_capture(aux_hidden_states_layers)

    @torch.no_grad
    def _extend(
        self,
        reqs,
        capture_aux_hidden_states: bool = True,
        return_last_hidden_states: bool = False,
        return_logits: bool = False,
    ):
        # set the logits processor for the model runner
        for name, module in self.model_runner.model.named_modules():
            if isinstance(module, LogitsProcessorForEAGLE3):
                module.return_last_hidden_states = return_last_hidden_states
                module.return_logits = return_logits

        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self._maybe_prepare_mlp_sync_batch(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        eagle3_output, _ = self.model_runner.forward(forward_batch)

        aux_hidden_states_list = None
        input_lens = [len(req.origin_input_ids) for req in reqs]

        if return_logits:
            logits = torch.split(eagle3_output.logits, input_lens, dim=0)
        else:
            logits = [None] * len(reqs)

        if capture_aux_hidden_states:
            aux_hidden_states_list = torch.split(
                eagle3_output.aux_hidden_states, input_lens, dim=0
            )
        else:
            aux_hidden_states_list = [None] * len(reqs)

        if return_last_hidden_states:
            last_hidden_states = torch.split(
                eagle3_output.last_hidden_states, input_lens, dim=0
            )
        else:
            last_hidden_states = [None] * len(reqs)

        # TODO: can we not clear?
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        mm_utils.embedding_cache.clear()
        return logits, aux_hidden_states_list, last_hidden_states

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch):
        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

    def extend(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
    ):
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids = torch.split(input_ids, 1, dim=0)
            attention_mask = torch.split(attention_mask, 1, dim=0)
            loss_mask = torch.split(loss_mask, 1, dim=0)

        for idx, (input_id_, attention_mask_, loss_mask_) in enumerate(
            zip(
                input_ids,
                attention_mask,
                loss_mask,
            )
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_id_.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            data_cache.append([input_id_, attention_mask_, loss_mask_])
            reqs.append(req)

        logits_list, aux_hidden_states_list, last_hidden_states_list = self._extend(
            reqs,
            capture_aux_hidden_states=True,
            return_last_hidden_states=return_last_hidden_states,
            return_logits=return_logits,
        )

        return data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get M-RoPE position indices for VLM models like Qwen2.5-VL.

        This is a wrapper around MRotaryEmbedding.get_rope_index that uses
        the VLM-specific attributes initialized from hf_config.

        Args:
            input_ids: (batch_size, seq_len) input token IDs
            image_grid_thw: (num_images, 3) image grid dimensions (t, h, w)
            video_grid_thw: (num_videos, 3) video grid dimensions (t, h, w)
            second_per_grid_ts: Optional temporal information for videos
            attention_mask: (batch_size, seq_len) attention mask

        Returns:
            position_ids: (3, batch_size, seq_len) M-RoPE position IDs
            rope_deltas: Optional position deltas for incremental decoding
        """
        if not self.is_vlm:
            raise ValueError("get_rope_index is only available for VLM models")

        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        position_ids, rope_deltas = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            tokens_per_second=self.tokens_per_second,
        )

        return position_ids, rope_deltas

    def extend_vlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        pixel_values: Optional[List[torch.Tensor]] = None,
        image_grid_thw: Optional[List[torch.Tensor]] = None,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len) or List of (1, seq_len) tensors
            attention_mask: (batch_size, seq_len) or List of (1, seq_len) tensors
            loss_mask: (batch_size, seq_len) or List of (1, seq_len) tensors
            pixel_values: List of pixel_values tensors, one per sample in batch
            image_grid_thw: List of image_grid_thw tensors, one per sample in batch
        """
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs, data_cache = [], []

        # Split tensors if needed
        if isinstance(input_ids, torch.Tensor):
            batch_size = input_ids.shape[0]
            input_ids = torch.split(input_ids, 1, dim=0)
            attention_mask = torch.split(attention_mask, 1, dim=0)
            loss_mask = torch.split(loss_mask, 1, dim=0)
        else:
            batch_size = len(input_ids)
        # Process image_grid_thw - convert to list if needed
        if image_grid_thw is None:
            image_grid_thw = [None] * batch_size
        elif not isinstance(image_grid_thw, (list, tuple)):
            image_grid_thw = [image_grid_thw]

        # pixel_values is a single 2D tensor (total_patches, patch_dim) for Qwen2.5-VL
        # We need to track offset and slice it based on image_grid_thw for each sample
        pixel_values_offset = 0  # Track current offset in pixel_values

        for idx, (input_id_, attention_mask_, loss_mask_, image_grid_thw_) in enumerate(
            zip(
                input_ids,
                attention_mask,
                loss_mask,
                image_grid_thw,
            )
        ):
            # Compute num_patches for this sample from image_grid_thw_
            # image_grid_thw_: (num_images, 3) where each row is (t, h, w)
            if image_grid_thw_ is not None:
                # Ensure image_grid_thw_ is 2D: (num_images, 3)
                if image_grid_thw_.dim() == 1:
                    image_grid_thw_ = image_grid_thw_.unsqueeze(0)  # (3,) -> (1, 3)
                elif image_grid_thw_.dim() == 0:
                    raise ValueError(
                        f"image_grid_thw_ is 0-dim tensor, expected at least 1D. Value: {image_grid_thw_}"
                    )

                # Calculate num_patches for this sample: sum(t * h * w) for all images
                num_patches = (
                    (
                        image_grid_thw_[:, 0]
                        * image_grid_thw_[:, 1]
                        * image_grid_thw_[:, 2]
                    )
                    .sum()
                    .item()
                )
                num_patches = int(num_patches)

                # Slice pixel_values for this sample
                pixel_value_ = pixel_values[
                    pixel_values_offset : pixel_values_offset + num_patches
                ]
                pixel_values_offset += num_patches
            else:
                pixel_value_ = None
                num_patches = 0

            # Compute mrope positions for VLM models (e.g., Qwen2.5-VL)
            input_id_flat = input_id_.view(-1)

            # Count image tokens
            num_img_tokens = (input_id_flat == self.image_token_id).sum().item()
            # print(f"[extend_vlm] num_img_tokens in input_ids: {num_img_tokens}")

            mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=self.spatial_merge_size,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type=self.model_type,
                input_ids=input_id_flat.unsqueeze(0),
                image_grid_thw=image_grid_thw_,
                tokens_per_second=self.tokens_per_second,
            )

            offset = BaseMultimodalProcessor.get_mm_items_offset(
                input_id_flat, self.image_token_id
            )
            mm_item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=pixel_value_,  # torch.Tensor: (num_patches, patch_dim)
                pad_value=self.image_token_id,  # Required for placeholder tensor creation
                offsets=offset,  # List of (start, end) tuples
            )
            mm_item.set("image_grid_thw", image_grid_thw_)
            mm_inputs = MultimodalInputs(
                mm_items=[mm_item],
                im_token_id=self.image_token_id,
                im_start_id=self.vision_start_token_id,
                im_end_id=self.vision_end_token_id,
                mrope_positions=(
                    mrope_positions.squeeze(1) if mrope_positions is not None else None
                ),
                mrope_position_delta=mrope_position_delta,
            )
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_id_.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.multimodal_inputs = mm_inputs
            data_cache.append([input_id_, attention_mask_, loss_mask_])
            reqs.append(req)

        logits_list, aux_hidden_states_list, last_hidden_states_list = self._extend(
            reqs,
            capture_aux_hidden_states=True,
            return_last_hidden_states=return_last_hidden_states,
            return_logits=return_logits,
        )

        return data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
    ) -> Eagle3TargetOutput:
        """
        return:
            data_for_draft: List[Dict[str, torch.Tensor]] of draft_batch_size, draft_micro_batch_size = 1
                - input_ids: (1, seq_len)
                - attention_mask: (1, seq_len)
                - loss_mask: (1, seq_len)
                - target: (1, seq_len, vocab_size) or (1, seq_len, hidden_size)
                - hidden_states: (1, seq_len, hidden_size)
                - pixel_values: (patch_len, patch_width)
                - image_grid_thw (batch_size, 3)
        """
        if is_vlm:
            data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list = (
                self.extend_vlm(
                    input_ids,
                    attention_mask,
                    loss_mask,
                    return_last_hidden_states=False,
                    return_logits=True,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            )
        else:
            data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list = (
                self.extend(
                    input_ids,
                    attention_mask,
                    loss_mask,
                    return_last_hidden_states=False,
                    return_logits=True,
                )
            )
        aux_hidden_states_out = []
        target_out = []
        loss_mask_out = []
        input_ids_out = []
        last_hidden_states_out = []

        for idx, (data, logits, aux_hidden_states, last_hidden_states) in enumerate(
            zip(
                data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list
            )
        ):
            aux_hidden_states_out.append(aux_hidden_states.unsqueeze(0))
            loss_mask_out.append(data[2])
            input_ids_out.append(data[0])

            # when generating hidden states for offline training, we don't compute logits and only keep the last_hidden_states
            # when training online, we don't keep the last_hidden_states and only keep the logits
            if logits is not None:
                target_out.append(logits.unsqueeze(0))
            else:
                target_out.append(None)

            if last_hidden_states is not None:
                last_hidden_states_out.append(last_hidden_states.unsqueeze(0))
            else:
                last_hidden_states_out.append(None)

        aux_hidden_states_out = torch.cat(aux_hidden_states_out, dim=0)

        loss_mask_out = torch.cat(loss_mask_out, dim=0)
        input_ids_out = torch.cat(input_ids_out, dim=0)

        if target_out[0] is not None:
            target_out = torch.cat(target_out, dim=0)
        else:
            target_out = None

        if last_hidden_states_out[0] is not None:
            last_hidden_states_out = torch.cat(last_hidden_states_out, dim=0)
        else:
            last_hidden_states_out = None

        target_out = padding(target_out, left=False)
        input_ids_out = padding(input_ids_out, left=False)
        loss_mask_out = loss_mask_out[..., None]

        return Eagle3TargetOutput(
            hidden_states=aux_hidden_states_out,
            target=target_out,
            loss_mask=loss_mask_out,
            input_ids=input_ids_out,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states_out,
        )


class CustomEagle3TargetModel(Eagle3TargetModel):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "CustomEagle3TargetModel":
        from specforge.modeling.auto import AutoDistributedTargetModel

        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            device=device,
            **kwargs,
        )
        return cls(target_model)

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            layers_to_output_hidden_states=self.aux_hidden_states_layers,
            use_cache=False,
        )

        # For custom backends, the model implementation is responsible for only
        # returning the requested layers in `outputs.hidden_states`.
        hidden_states = torch.cat(outputs.hidden_states, dim=-1)

        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(target.device)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


def get_eagle3_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Eagle3TargetModel:
    if backend == "sglang":
        return SGLangEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "custom":
        return CustomEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
