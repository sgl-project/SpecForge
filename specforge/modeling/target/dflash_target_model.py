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
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoConfig, AutoModelForCausalLM

from specforge.distributed import get_tp_group

from .sglang_backend import SGLangRunner, wrap_eagle3_logits_processors_in_module

QWEN3_VL_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe"}


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor  # [batch, seq_len, hidden_size]
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]
    position_ids: Optional[torch.Tensor] = None  # [batch, seq_len] or [3, batch, seq_len]


class DFlashTargetModel(ABC):
    """
    Abstract base class for DFlash target model backend.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training."""

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids


class SGLangDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model_runner: SGLangRunner, hf_config=None):
        super().__init__()
        self.model_runner = model_runner
        self.hf_config = hf_config
        self._init_vlm_attributes()

    def _init_vlm_attributes(self):
        if self.hf_config is None:
            self.is_vlm = False
            return

        self.is_vlm = hasattr(self.hf_config, "vision_config")
        if not self.is_vlm:
            return

        init_mm_embedding_cache(1024 * 1024 * 512)
        self.model_type = getattr(self.hf_config, "model_type", None)
        vision_config = self.hf_config.vision_config
        self.spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        self.tokens_per_second = getattr(vision_config, "tokens_per_second", None)
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
    ) -> "SGLangDFlashTargetModel":
        tp_size = dist.get_world_size(get_tp_group())
        dtype_arg = torch_dtype if torch_dtype is not None else "auto"
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype_arg,
            enable_return_hidden_states=True,  # Critical for DFlash
            disable_cuda_graph=True,
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
            is_draft_worker=False,
        )
        wrap_eagle3_logits_processors_in_module(
            model_runner.model, return_full_logits=False
        )
        hf_config = getattr(model_config, "hf_config", None)
        return cls(model_runner, hf_config=hf_config)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)
        if hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            self.model_runner.model.set_eagle3_layers_to_capture(layer_ids)

    @torch.no_grad
    def _extend(self, reqs):
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

        if require_mlp_sync(self.model_runner.server_args):
            prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                attn_cp_size=getattr(self.model_runner.server_args, "attn_cp_size", 1),
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        output = self.model_runner.forward(forward_batch)
        if hasattr(output, "logits_output"):
            output = output.logits_output

        input_lens = [len(req.origin_input_ids) for req in reqs]
        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            hidden_states_list = torch.split(
                output.aux_hidden_states, input_lens, dim=0
            )
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden_states_list = torch.split(output.hidden_states, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")

        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

        return hidden_states_list

    @staticmethod
    def _split_per_sample_tensor(
        tensor: Optional[torch.Tensor], batch_size: int, name: str
    ) -> List[Optional[torch.Tensor]]:
        if tensor is None:
            return [None] * batch_size
        if isinstance(tensor, (list, tuple)):
            return list(tensor)
        if not isinstance(tensor, torch.Tensor):
            return [tensor] * batch_size
        if batch_size == 1:
            return [tensor.squeeze(0) if tensor.dim() > 1 and tensor.shape[0] == 1 else tensor]
        if tensor.dim() > 0 and tensor.shape[0] == batch_size:
            return [x.squeeze(0) for x in torch.split(tensor, 1, dim=0)]
        raise ValueError(
            f"Cannot split {name} with shape {tuple(tensor.shape)} across batch size {batch_size}."
        )

    @staticmethod
    def _normalize_grid_thw(
        grid_thw: Optional[torch.Tensor], name: str
    ) -> Optional[torch.Tensor]:
        if grid_thw is None:
            return None
        if grid_thw.dim() == 1:
            return grid_thw.unsqueeze(0)
        if grid_thw.dim() == 2:
            return grid_thw
        raise ValueError(
            f"{name} must be a 1D or 2D tensor, got shape {tuple(grid_thw.shape)}"
        )

    @staticmethod
    def _count_mm_patches(grid_thw: Optional[torch.Tensor]) -> int:
        if grid_thw is None:
            return 0
        return int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())

    def _build_mm_inputs(
        self,
        input_id_flat: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        second_per_grid_ts: Optional[torch.Tensor],
    ) -> Tuple[Optional[MultimodalInputs], Optional[torch.Tensor]]:
        mm_items = []

        if pixel_values is not None:
            image_offsets = BaseMultimodalProcessor.get_mm_items_offset(
                input_id_flat, self.image_token_id
            )
            image_item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=pixel_values,
                pad_value=self.image_token_id,
                offsets=image_offsets,
            )
            if image_grid_thw is not None:
                image_item.set("image_grid_thw", image_grid_thw.cpu())
            image_item.set_pad_value()
            mm_items.append(image_item)

        if pixel_values_videos is not None:
            video_offsets = BaseMultimodalProcessor.get_mm_items_offset(
                input_id_flat, self.video_token_id
            )
            video_item = MultimodalDataItem(
                modality=Modality.VIDEO,
                feature=pixel_values_videos,
                pad_value=self.video_token_id,
                offsets=video_offsets,
            )
            if video_grid_thw is not None:
                video_item.set("video_grid_thw", video_grid_thw.cpu())
            if second_per_grid_ts is not None:
                video_item.set("second_per_grid_ts", second_per_grid_ts.cpu())
            video_item.set_pad_value()
            mm_items.append(video_item)

        if not mm_items:
            return None, None

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            input_ids=input_id_flat.unsqueeze(0).cpu(),
            image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
            second_per_grid_ts=(
                second_per_grid_ts.cpu() if second_per_grid_ts is not None else None
            ),
            tokens_per_second=self.tokens_per_second,
        )
        mm_inputs = MultimodalInputs(
            mm_items=mm_items,
            im_token_id=self.image_token_id,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            video_token_id=self.video_token_id,
            mrope_positions=(
                mrope_positions.squeeze(1) if mrope_positions is not None else None
            ),
            mrope_position_delta=mrope_position_delta,
        )
        return mm_inputs, mm_inputs.mrope_positions

    @torch.no_grad()
    def _extend_vlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], list, Optional[torch.Tensor]]:
        mm_utils.embedding_cache.clear()
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs, data_cache, position_ids_list = [], [], []

        if isinstance(input_ids, torch.Tensor):
            batch_size = input_ids.shape[0]
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)
        else:
            batch_size = len(input_ids)
            input_ids_list = input_ids
            attn_mask_list = attention_mask
            loss_mask_list = loss_mask

        image_grid_thw_list = self._split_per_sample_tensor(
            image_grid_thw, batch_size, "image_grid_thw"
        )
        video_grid_thw_list = self._split_per_sample_tensor(
            video_grid_thw, batch_size, "video_grid_thw"
        )
        second_per_grid_ts_list = self._split_per_sample_tensor(
            second_per_grid_ts, batch_size, "second_per_grid_ts"
        )

        image_offset = 0
        video_offset = 0
        pattern = None

        for idx, (
            curr_ids,
            curr_attn,
            curr_loss,
            curr_image_grid,
            curr_video_grid,
            curr_second_per_grid,
        ) in enumerate(
            zip(
                input_ids_list,
                attn_mask_list,
                loss_mask_list,
                image_grid_thw_list,
                video_grid_thw_list,
                second_per_grid_ts_list,
            )
        ):
            curr_image_grid = self._normalize_grid_thw(curr_image_grid, "image_grid_thw")
            curr_video_grid = self._normalize_grid_thw(curr_video_grid, "video_grid_thw")

            image_patches = self._count_mm_patches(curr_image_grid)
            video_patches = self._count_mm_patches(curr_video_grid)

            curr_pixel_values = None
            if pixel_values is not None and image_patches > 0:
                curr_pixel_values = pixel_values[image_offset : image_offset + image_patches]
                image_offset += image_patches

            curr_pixel_values_videos = None
            if pixel_values_videos is not None and video_patches > 0:
                curr_pixel_values_videos = pixel_values_videos[
                    video_offset : video_offset + video_patches
                ]
                video_offset += video_patches

            input_id_flat = curr_ids.view(-1)
            mm_inputs, position_ids = self._build_mm_inputs(
                input_id_flat=input_id_flat,
                pixel_values=curr_pixel_values,
                pixel_values_videos=curr_pixel_values_videos,
                image_grid_thw=curr_image_grid,
                video_grid_thw=curr_video_grid,
                second_per_grid_ts=curr_second_per_grid,
            )
            input_id_list = input_id_flat.tolist()
            if mm_inputs is not None:
                if pattern is None:
                    from sglang.srt.managers.mm_utils import (
                        MultiModalityDataPaddingPatternMultimodalTokens,
                    )

                    pattern = MultiModalityDataPaddingPatternMultimodalTokens()
                input_id_list = pattern.pad_input_tokens(input_id_list, mm_inputs)

            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_id_list,
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            if mm_inputs is not None:
                req.multimodal_inputs = mm_inputs

            data_cache.append((curr_ids, curr_attn, curr_loss))
            position_ids_list.append(position_ids)
            reqs.append(req)

        hidden_states_list = self._extend(reqs)

        position_ids = None
        if position_ids_list and all(pos is not None for pos in position_ids_list):
            position_ids = torch.stack(position_ids_list, dim=1)

        return hidden_states_list, data_cache, position_ids

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> DFlashTargetOutput:
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = [], []
        position_ids = None

        use_multimodal = any(
            item is not None
            for item in (
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
            )
        )

        if use_multimodal:
            if not self.is_vlm:
                raise ValueError(
                    "Multimodal inputs were provided to a non-VLM SGLang target model."
                )
            hidden_states_list, data_cache, position_ids = self._extend_vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            if isinstance(input_ids, torch.Tensor):
                input_ids_list = torch.split(input_ids, 1, dim=0)
                attn_mask_list = torch.split(attention_mask, 1, dim=0)
                loss_mask_list = torch.split(loss_mask, 1, dim=0)

            for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
                zip(input_ids_list, attn_mask_list, loss_mask_list)
            ):
                req = Req(
                    rid=str(idx),
                    origin_input_text="",
                    origin_input_ids=curr_ids.view(-1).tolist(),
                    sampling_params=sampling_params,
                )
                req.fill_ids = req.origin_input_ids
                req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
                data_cache.append((curr_ids, curr_attn, curr_loss))
                reqs.append(req)

            hidden_states_list = self._extend(reqs)

        # Stack back to batch
        hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states_list], dim=0)
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
        )


class HFDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model: nn.Module, model_type: Optional[str] = None):
        super().__init__()
        self.model = model
        self.model_type = model_type

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetModel":
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        model_type = getattr(hf_config, "model_type", None)

        if model_type in QWEN3_VL_MODEL_TYPES:
            if model_type == "qwen3_vl":
                try:
                    from transformers import Qwen3VLForConditionalGeneration
                except ImportError as exc:
                    raise ImportError(
                        "Qwen3VLForConditionalGeneration is unavailable. "
                        "Please upgrade transformers to a version with qwen3_vl support."
                    ) from exc

                model_cls = Qwen3VLForConditionalGeneration
            else:
                try:
                    from transformers import Qwen3VLMoeForConditionalGeneration
                except ImportError as exc:
                    raise ImportError(
                        "Qwen3VLMoeForConditionalGeneration is unavailable. "
                        "Please upgrade transformers to a version with qwen3_vl_moe support."
                    ) from exc

                model_cls = Qwen3VLMoeForConditionalGeneration
            target_model = model_cls.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs,
            ).eval()
        else:
            target_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                output_hidden_states=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            ).eval()

        if device:
            target_model = target_model.to(device)

        return cls(target_model, model_type=model_type)

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> DFlashTargetOutput:
        target_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if self.model_type in QWEN3_VL_MODEL_TYPES:
            target_kwargs.update(
                {
                    "pixel_values": pixel_values,
                    "pixel_values_videos": pixel_values_videos,
                    "image_grid_thw": image_grid_thw,
                    "video_grid_thw": video_grid_thw,
                }
            )

        filtered_target_kwargs = {}
        for key, value in target_kwargs.items():
            if key in {
                "input_ids",
                "attention_mask",
                "output_hidden_states",
                "use_cache",
            } or value is not None:
                filtered_target_kwargs[key] = value

        outputs = self.model(**filtered_target_kwargs)
        if outputs.hidden_states is None:
            raise ValueError(
                "Target model did not return hidden states. Ensure output_hidden_states=True is supported."
            )

        position_ids = None
        if self.model_type in QWEN3_VL_MODEL_TYPES:
            target_inner_model = getattr(self.model, "model", None)
            if target_inner_model is not None and hasattr(
                target_inner_model, "get_rope_index"
            ):
                rope_kwargs = {
                    "input_ids": input_ids,
                    "image_grid_thw": image_grid_thw,
                    "attention_mask": attention_mask,
                }
                if video_grid_thw is not None:
                    rope_kwargs["video_grid_thw"] = video_grid_thw
                if second_per_grid_ts is not None:
                    rope_kwargs["second_per_grid_ts"] = second_per_grid_ts

                filtered_rope_kwargs = {
                    key: value for key, value in rope_kwargs.items() if value is not None
                }
                position_ids, _ = target_inner_model.get_rope_index(
                    **filtered_rope_kwargs
                )

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        offset = 1
        selected = []
        if self.capture_layer_ids is not None:
            for idx in self.capture_layer_ids:
                selected.append(outputs.hidden_states[idx + offset])
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = outputs.hidden_states[-1]

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
        )


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetModel:
    if backend == "sglang":
        return SGLangDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
