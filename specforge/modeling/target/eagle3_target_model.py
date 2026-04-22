from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sglang.srt.managers.mm_utils as mm_utils
import torch
import torch.distributed as dist
import torch.nn as nn
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    init_mm_embedding_cache,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)

# - prepare_mlp_sync_batch_raw is now a module-level function, not a Scheduler method
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
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
    target_in_draft_mask: Optional[torch.Tensor] = None  # pre-computed bool mask: whether target argmax is in draft vocab
    pre_projected: bool = False  # whether target logits are already projected to draft_vocab_size


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
        # NOTE: sglang 0.5.9 requires dtype to be non-None
        # If torch_dtype is None, use "auto" to let sglang decide the dtype
        dtype_arg = torch_dtype if torch_dtype is not None else "auto"
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype_arg,
            enable_return_hidden_states=True,
            disable_cuda_graph=True,  # we use piecewise cuda graph for prefill instead
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)
        # - Added is_draft_worker=False parameter (new in 0.5.9)
        # - Other new parameters (dp_rank, attn_cp_rank, moe_dp_rank, etc.) use default values
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

        # Get hf_config from model_config for VLM attributes
        hf_config = getattr(model_config, "hf_config", None)

        return cls(model_runner, hf_config=hf_config)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        model = self.model_runner.model
        if hasattr(model, "set_eagle3_layers_to_capture"):
            model.set_eagle3_layers_to_capture(aux_hidden_states_layers)
        else:
            # Fallback for models that don't natively support Eagle3 layer capture
            # (e.g., Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM).
            # We monkey-patch the forward methods to inject aux_hidden_states capture logic.
            print(
                f"[Eagle3] Model {type(model).__name__} does not have "
                f"set_eagle3_layers_to_capture, applying monkey-patch..."
            )
            self._monkey_patch_eagle3_layer_capture(model, aux_hidden_states_layers)

    def set_vocab_mapping(self, t2d: torch.Tensor) -> None:
        """
        Store the target-to-draft vocab mapping for early vocab projection.

        When set, the logits processor will project full-vocab logits to draft_vocab
        *inside* the chunked computation loop, so the full-vocab tensor is never
        materialised across all tokens simultaneously.

        Memory improvement:
        - Without: chunked logits are computed then torch.cat → peak = 2 × (tokens × full_vocab)
        - With:    each chunk is projected immediately → peak = 1 chunk × full_vocab + tokens × draft_vocab
        - For 20K tokens, vocab 248K, draft 32K: 19 GiB → 2.2 GiB (8.6× reduction)

        Args:
            t2d: Bool tensor of shape (target_vocab_size,) indicating which target vocab
                 tokens are in the draft vocabulary.
        """
        self._t2d = t2d.cuda()

        # Set on the module-level global (backward compatibility)
        from .sglang_backend import utils as sglang_utils
        sglang_utils._T2D_MAPPING = self._t2d

        # Also set directly on each LogitsProcessorForEAGLE3 instance — this is the
        # primary mechanism.  Instance-level takes priority over the module-level
        # global inside replaced_logits_processor_forward_for_eagle3(), eliminating
        # any risk of the global variable not being visible.
        n_set = 0
        for _name, module in self.model_runner.model.named_modules():
            if isinstance(module, LogitsProcessorForEAGLE3):
                module.t2d_mapping = self._t2d
                n_set += 1
        print(
            f"[Eagle3] Vocab mapping set: target_vocab={t2d.shape[0]}, "
            f"draft_vocab={t2d.sum().item()}, "
            f"set on {n_set} LogitsProcessorForEAGLE3 instance(s) + module-level global"
        )

    def set_logits_chunk_size(self, chunk_size: int) -> None:
        """
        Set the chunk size for chunked logits computation.

        When chunk_size > 0 and total tokens exceed this value, logits are computed
        in chunks to reduce peak all_gather memory from (total_tokens × vocab_size)
        to (chunk_size × vocab_size).

        Args:
            chunk_size: Number of tokens per chunk. 0 = disable chunking.
        """
        # Set on the module-level global (backward compatibility)
        from .sglang_backend import utils as sglang_utils
        sglang_utils.LOGITS_CHUNK_SIZE = chunk_size

        # Also set directly on each LogitsProcessorForEAGLE3 instance
        n_set = 0
        for _name, module in self.model_runner.model.named_modules():
            if isinstance(module, LogitsProcessorForEAGLE3):
                module.logits_chunk_size = chunk_size
                n_set += 1
        print(
            f"[Eagle3] Logits chunk size set to {chunk_size} "
            f"({'enabled' if chunk_size > 0 else 'disabled'}), "
            f"set on {n_set} LogitsProcessorForEAGLE3 instance(s) + module-level global"
        )

    def _monkey_patch_eagle3_layer_capture(
        self, model: nn.Module, layer_ids: Optional[List[int]] = None
    ) -> None:
        """
        Monkey-patch Eagle3 aux_hidden_states capture logic onto models that don't
        natively support it (e.g., Qwen3_5MoeForConditionalGeneration).

        This replicates the behavior of Qwen3MoeForCausalLM.set_eagle3_layers_to_capture
        and Qwen3MoeModel.forward's aux_hidden_states collection logic.

        Model hierarchy handled:
        - VLM: Qwen3_5MoeForConditionalGeneration
            - self.model = Qwen3_5MoeForCausalLM (inherits Qwen3_5ForCausalLM)
                - self.layers = decoder layers
                - self.embed_tokens, self.norm, etc.
        - Non-VLM CausalLM without Eagle3 support (similar flat structure)
        """
        import types

        # --- Step 1: Find the language model (the one with self.layers) ---
        language_model = self._find_language_model(model)
        if language_model is None:
            raise ValueError(
                f"Cannot find language model with 'layers' attribute in {type(model).__name__}. "
                f"Cannot apply Eagle3 monkey-patch."
            )

        # --- Step 2: Determine layer IDs to capture ---
        num_layers = len(language_model.layers)
        if layer_ids is None:
            # Default Eagle3 layers: low, mid, high
            # In Qwen3MoeForCausalLM.set_eagle3_layers_to_capture, the default is
            # [2, num_layers//2, num_layers-3] with +1 offset because it captures
            # at the START of the marked layer (i.e., previous layer's output).
            # Here we capture AFTER the marked layer completes, so we use the
            # actual target layer indices directly (equivalent to layer_ids=[1, num_layers//2-1, num_layers-4]).
            capture_layer_ids = [1, num_layers // 2 - 1, num_layers - 4]
        else:
            # User-provided layer_ids are 0-indexed target layers.
            # In Qwen3MoeForCausalLM, layer_ids are offset by +1 because capture
            # happens at the start of the next layer. Here we capture after the
            # layer completes, so we use layer_ids directly.
            capture_layer_ids = list(layer_ids)

        # --- Step 3: Set _is_layer_to_capture on target decoder layers ---
        language_model.layers_to_capture = capture_layer_ids
        for layer_id in capture_layer_ids:
            if 0 <= layer_id < num_layers:
                setattr(language_model.layers[layer_id], "_is_layer_to_capture", True)
            else:
                raise ValueError(
                    f"Layer index {layer_id} out of bounds for model with {num_layers} layers."
                )

        print(
            f"[Eagle3] Set _is_layer_to_capture on layers {capture_layer_ids} "
            f"(total {num_layers} layers)"
        )

        # --- Step 4: Patch the language model's forward to collect aux_hidden_states ---
        # Note: Qwen3_5AttentionDecoderLayer/Qwen3_5LinearDecoderLayer do NOT handle
        # `captured_last_layer_outputs` parameter (unlike Qwen3MoeDecoderLayer).
        # So we capture hidden states manually after each marked layer executes.
        #
        # In Qwen3MoeModel, `captured_last_layer_outputs` captures the residual at the
        # start of the marked layer (i.e., the previous layer's output via prepare_attn).
        # Here we capture the residual after the marked layer completes (post MLP + residual).
        # To get the same effective layer outputs, we use the target layer indices directly
        # without the +1 offset that Qwen3MoeForCausalLM applies.
        original_lm_forward = language_model.forward

        # Try to import expert_distribution recorder (optional, for MoE expert stats).
        # Different SGLang versions use different module paths. Resolve once at patch time.
        _get_recorder = None
        for _mod_path in (
            "sglang.srt.eplb.expert_distribution",
            "sglang.srt.managers.expert_distribution",
        ):
            try:
                _mod = __import__(_mod_path, fromlist=["get_global_expert_distribution_recorder"])
                _get_recorder = _mod.get_global_expert_distribution_recorder
                break
            except (ImportError, ModuleNotFoundError):
                continue

        if _get_recorder is None:
            print(
                "[Eagle3] Warning: expert_distribution module not found in SGLang. "
                "Expert distribution recording will be disabled."
            )

        def patched_lm_forward(
            self_lm,
            input_ids=None,
            positions=None,
            forward_batch=None,
            input_embeds=None,
            pp_proxy_tensors=None,
            input_deepstack_embeds=None,
            **kwargs,
        ):
            """Patched forward that captures aux_hidden_states from marked layers."""
            # Initialize hidden states
            if self_lm.pp_group.is_first_rank:
                if input_embeds is None:
                    hidden_states = self_lm.embed_tokens(input_ids)
                else:
                    hidden_states = input_embeds
                residual = None
            else:
                assert pp_proxy_tensors is not None
                hidden_states = pp_proxy_tensors["hidden_states"]
                residual = pp_proxy_tensors["residual"]

            # Pass through decoder layers with aux_hidden_states capture
            aux_hidden_states = []
            for layer_idx in range(len(self_lm.layers)):
                layer = self_lm.layers[layer_idx]

                # Use expert distribution recorder context if available
                if _get_recorder is not None:
                    with _get_recorder().with_current_layer(layer_idx):
                        hidden_states, residual = layer(
                            positions=positions,
                            hidden_states=hidden_states,
                            residual=residual,
                            forward_batch=forward_batch,
                        )
                else:
                    hidden_states, residual = layer(
                        positions=positions,
                        hidden_states=hidden_states,
                        residual=residual,
                        forward_batch=forward_batch,
                    )

                # Capture aux_hidden_states for Eagle3 from marked layers.
                # We capture `residual` which contains the full hidden states
                # (hidden + residual connection) before the next layer's RMSNorm.
                # This is equivalent to what Qwen3MoeModel captures via
                # prepare_attn_and_capture_last_layer_outputs in the next layer.
                if getattr(layer, "_is_layer_to_capture", False):
                    # Clone to avoid in-place modification by subsequent layers
                    aux_hidden_states.append(residual.clone())

                # Process deepstack embeddings if provided
                if (
                    input_deepstack_embeds is not None
                    and input_deepstack_embeds.numel() > 0
                    and layer_idx < 3
                ):
                    sep = self_lm.hidden_size * layer_idx
                    hidden_states.add_(
                        input_deepstack_embeds[:, sep : sep + self_lm.hidden_size]
                    )

            # Return intermediate tensors for pipeline parallelism
            if not self_lm.pp_group.is_last_rank:
                from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

                return PPProxyTensors(
                    {
                        "hidden_states": hidden_states,
                        "residual": residual,
                    }
                )

            # Apply final normalization
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self_lm.norm(hidden_states)
                else:
                    hidden_states, _ = self_lm.norm(hidden_states, residual)

            if len(aux_hidden_states) == 0:
                return hidden_states

            return hidden_states, aux_hidden_states

        # Bind the patched forward to the language model instance
        language_model.forward = types.MethodType(patched_lm_forward, language_model)

        # --- Step 5: Handle the outer model (VLM or CausalLM wrapper) ---
        # For VLM models like Qwen3_5MoeForConditionalGeneration, the outer forward
        # calls general_mm_embed_routine which calls language_model(...).
        # The language_model now returns (hidden_states, aux_hidden_states) tuple.
        # We need the outer forward to unpack this and pass aux_hidden_states to logits_processor.

        # Check if model is a VLM wrapper (has self.model which is the language model)
        if hasattr(model, "model") and model.model is language_model:
            # VLM model: patch the outer forward
            self._patch_vlm_forward_for_eagle3(model, language_model)
        elif model is language_model:
            # Direct CausalLM model (no VLM wrapper): need to handle aux_hidden_states
            # in the logits processing step. Set capture flag.
            model.capture_aux_hidden_states = True

        print(f"[Eagle3] Monkey-patch applied successfully to {type(model).__name__}")

    def _find_language_model(self, model: nn.Module) -> Optional[nn.Module]:
        """
        Find the language model component that has self.layers (decoder layers).

        Handles various model hierarchies:
        - VLM: model.model (language model with layers)
        - CausalLM with inner model: model.model (inner model with layers)
        - Flat CausalLM: model itself has layers
        """
        # Check if model itself has layers
        if hasattr(model, "layers"):
            return model

        # Check model.model (common for VLM and some CausalLM)
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "layers"):
                return inner
            # Check model.model.model (deeper nesting)
            if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                return inner.model

        # Check model.language_model
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "layers"):
                return lm
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model

        return None

    def _patch_vlm_forward_for_eagle3(
        self, vlm_model: nn.Module, language_model: nn.Module
    ) -> None:
        """
        Patch the VLM outer model's forward to handle aux_hidden_states from
        the patched language model.

        For Qwen3_5MoeForConditionalGeneration, the forward calls
        general_mm_embed_routine which calls language_model.forward().
        After patching, language_model.forward() returns (hidden_states, aux_hidden_states).
        We need to unpack this and pass aux_hidden_states to logits_processor.
        """
        import types

        original_vlm_forward = vlm_model.forward

        def patched_vlm_forward(
            self_vlm,
            input_ids,
            positions,
            forward_batch,
            get_embedding=False,
            pp_proxy_tensors=None,
        ):
            """Patched VLM forward that handles aux_hidden_states from language model."""
            from sglang.srt.models.qwen3_vl import general_mm_embed_routine

            if hasattr(self_vlm, "is_mrope_enabled") and self_vlm.is_mrope_enabled:
                positions = forward_batch.mrope_positions

            if not (
                forward_batch.forward_mode.is_decode()
                or not forward_batch.contains_image_inputs()
            ):
                if (
                    hasattr(self_vlm, "is_mrope_enabled")
                    and self_vlm.is_mrope_enabled
                ):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}"
                    )

            result = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self_vlm.model,
                multimodal_model=self_vlm,
                positions=positions,
                use_deepstack=getattr(self_vlm, "use_deepstack", {}),
                pp_proxy_tensors=pp_proxy_tensors,
            )

            # The patched language model may return (hidden_states, aux_hidden_states)
            aux_hidden_states = None
            if isinstance(result, tuple) and len(result) == 2:
                hidden_states, aux_hidden_states = result
            else:
                hidden_states = result

            if self_vlm.pp_group.is_last_rank:
                if not get_embedding:
                    return self_vlm.logits_processor(
                        input_ids,
                        hidden_states,
                        self_vlm.lm_head,
                        forward_batch,
                        aux_hidden_states,
                    )
                else:
                    return self_vlm.pooler(hidden_states, forward_batch)
            else:
                return hidden_states

        vlm_model.forward = types.MethodType(patched_vlm_forward, vlm_model)

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
        eagle3_output = self.model_runner.forward(forward_batch)
        aux_hidden_states_list = None
        input_lens = [len(req.origin_input_ids) for req in reqs]

        # Extract the logits output object (may carry pre-projection metadata)
        logits_output_obj = getattr(eagle3_output, "logits_output", eagle3_output)

        # Check if logits were pre-projected inside the logits processor
        logits_pre_projected = getattr(logits_output_obj, "logits_pre_projected", False)
        target_in_draft_mask_raw = getattr(logits_output_obj, "target_in_draft_mask", None)

        if return_logits:
            raw_logits = logits_output_obj.logits
            logits = torch.split(raw_logits, input_lens, dim=0)
        else:
            logits = [None] * len(reqs)

        # Split target_in_draft_mask per sample if present
        if target_in_draft_mask_raw is not None:
            target_in_draft_mask_list = torch.split(
                target_in_draft_mask_raw, input_lens, dim=0
            )
        else:
            target_in_draft_mask_list = [None] * len(reqs)

        if capture_aux_hidden_states:
            raw_aux_hidden_states = (
                logits_output_obj.aux_hidden_states
            )  # concat hidden shape: (total_tokens, H*3)
            aux_hidden_states_list = torch.split(
                raw_aux_hidden_states, input_lens, dim=0
            )
        else:
            aux_hidden_states_list = [None] * len(reqs)

        if return_last_hidden_states:
            last_hidden_states = torch.split(
                logits_output_obj.last_hidden_states, input_lens, dim=0
            )
        else:
            last_hidden_states = [None] * len(reqs)

        # TODO: can we not clear?
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        return (
            logits, aux_hidden_states_list, last_hidden_states,
            logits_pre_projected, target_in_draft_mask_list,
        )

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch):
        if require_mlp_sync(self.model_runner.server_args):
            # - Removed spec_algorithm and speculative_num_draft_tokens parameters
            # - Added attn_cp_size parameter
            # - Changed from Scheduler.prepare_mlp_sync_batch_raw to direct function call
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

        (
            logits_list, aux_hidden_states_list, last_hidden_states_list,
            logits_pre_projected, target_in_draft_mask_list,
        ) = self._extend(
            reqs,
            capture_aux_hidden_states=True,
            return_last_hidden_states=return_last_hidden_states,
            return_logits=return_logits,
        )

        return (
            data_cache, logits_list, aux_hidden_states_list,
            last_hidden_states_list, logits_pre_projected,
            target_in_draft_mask_list,
        )

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
        mm_utils.embedding_cache.clear()
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
                input_ids=input_id_flat.unsqueeze(0).cpu(),
                image_grid_thw=(
                    image_grid_thw_.cpu() if image_grid_thw_ is not None else None
                ),
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
            mm_item.set("image_grid_thw", image_grid_thw_.cpu())
            mm_item.set_pad_value()
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
            pattern = MultiModalityDataPaddingPatternMultimodalTokens()
            input_id_list = pattern.pad_input_tokens(
                input_id_.view(-1).tolist(), mm_inputs
            )
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_id_list,
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.multimodal_inputs = mm_inputs
            data_cache.append([input_id_, attention_mask_, loss_mask_])
            reqs.append(req)

        (
            logits_list, aux_hidden_states_list, last_hidden_states_list,
            logits_pre_projected, target_in_draft_mask_list,
        ) = self._extend(
            reqs,
            capture_aux_hidden_states=True,
            return_last_hidden_states=return_last_hidden_states,
            return_logits=return_logits,
        )

        return (
            data_cache, logits_list, aux_hidden_states_list,
            last_hidden_states_list, logits_pre_projected,
            target_in_draft_mask_list,
        )

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
        target_micro_batch_size: int = 0,
    ) -> Eagle3TargetOutput:
        """
        Generate Eagle3 training data from the target model.

        Memory optimization (v2): Early vocab projection is now done INSIDE the logits
        processor (in sglang_backend/utils.py), so the full-vocab logits tensor is never
        materialized across all tokens simultaneously. This is controlled by set_vocab_mapping()
        which sets _T2D_MAPPING on the utils module.

        With target_micro_batch_size > 0, samples are additionally processed in micro-batches
        for further memory savings on the KV cache side.

        Memory chain for a 20K-token sample on vocab_size=248K, draft_vocab=32K:
        - Without optimization: ~19 GiB (full-vocab logits for all tokens)
        - With logits-processor projection: ~2.2 GiB (chunk peak + projected logits)
        - With micro-batch + projection: same logits savings + KV cache savings

        Args:
            target_micro_batch_size: Number of samples per micro-batch for target inference.
                0 = process all samples at once (original behavior).
                1 = process one sample at a time (minimum memory usage, recommended for large vocab).
        """
        batch_size = input_ids.shape[0]
        use_micro_batch = target_micro_batch_size > 0

        if not use_micro_batch:
            # --- Process all samples at once ---
            # Note: even without micro-batching, early projection happens inside the
            # logits processor if _T2D_MAPPING is set (via set_vocab_mapping).
            if is_vlm:
                (
                    data_cache, logits_list, aux_hidden_states_list,
                    last_hidden_states_list, logits_pre_projected,
                    target_in_draft_mask_list,
                ) = self.extend_vlm(
                    input_ids,
                    attention_mask,
                    loss_mask,
                    return_last_hidden_states=False,
                    return_logits=True,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            else:
                (
                    data_cache, logits_list, aux_hidden_states_list,
                    last_hidden_states_list, logits_pre_projected,
                    target_in_draft_mask_list,
                ) = self.extend(
                    input_ids,
                    attention_mask,
                    loss_mask,
                    return_last_hidden_states=False,
                    return_logits=True,
                )

            if logits_pre_projected:
                # Logits are already projected to draft vocab inside logits processor
                return self._assemble_eagle3_output_projected(
                    data_cache, logits_list, aux_hidden_states_list,
                    target_in_draft_mask_list, attention_mask,
                )
            else:
                return self._assemble_eagle3_output(
                    data_cache, logits_list, aux_hidden_states_list,
                    last_hidden_states_list, attention_mask,
                )

        # --- Memory-optimized path: micro-batch ---
        mbs = target_micro_batch_size

        # Split inputs into micro-batches
        input_ids_splits = torch.split(input_ids, mbs, dim=0)
        attention_mask_splits = torch.split(attention_mask, mbs, dim=0)
        loss_mask_splits = torch.split(loss_mask, mbs, dim=0)

        # Handle VLM pixel_values and image_grid_thw splitting
        if is_vlm and image_grid_thw is not None:
            if isinstance(image_grid_thw, (list, tuple)):
                image_grid_thw_splits = [
                    image_grid_thw[i : i + mbs]
                    for i in range(0, len(image_grid_thw), mbs)
                ]
            else:
                image_grid_thw_splits = [None] * len(input_ids_splits)
            pixel_values_splits = self._split_pixel_values_by_grid_thw(
                pixel_values, image_grid_thw, mbs
            )
        else:
            image_grid_thw_splits = [None] * len(input_ids_splits)
            pixel_values_splits = [None] * len(input_ids_splits)

        # Process micro-batches
        all_aux_hidden_states = []
        all_projected_logits = []
        all_target_in_draft_mask = []
        all_input_ids = []
        all_loss_mask = []
        any_pre_projected = False

        for mb_idx, (ids_mb, attn_mb, lm_mb) in enumerate(
            zip(input_ids_splits, attention_mask_splits, loss_mask_splits)
        ):
            # Forward through target model
            if is_vlm:
                (
                    data_cache, logits_list, aux_hs_list, _,
                    logits_pre_projected, timd_list,
                ) = self.extend_vlm(
                    ids_mb, attn_mb, lm_mb,
                    return_last_hidden_states=False,
                    return_logits=True,
                    pixel_values=pixel_values_splits[mb_idx],
                    image_grid_thw=image_grid_thw_splits[mb_idx],
                )
            else:
                (
                    data_cache, logits_list, aux_hs_list, _,
                    logits_pre_projected, timd_list,
                ) = self.extend(
                    ids_mb, attn_mb, lm_mb,
                    return_last_hidden_states=False,
                    return_logits=True,
                )

            if logits_pre_projected:
                any_pre_projected = True

            # Collect per-sample results
            for i, (data, logits, aux_hs, timd) in enumerate(
                zip(data_cache, logits_list, aux_hs_list, timd_list)
            ):
                all_aux_hidden_states.append(aux_hs.unsqueeze(0))
                all_input_ids.append(data[0])
                all_loss_mask.append(data[2])

                if logits is not None:
                    # Determine if logits are actually projected by checking shape,
                    # NOT relying on the logits_pre_projected flag alone.
                    # This is the most reliable check: if the last dimension matches
                    # draft_vocab_size, it's projected; if it matches full vocab, it's not.
                    t2d = self._t2d
                    draft_vocab_size = int(t2d.sum().item()) if t2d is not None else None
                    actually_projected = (
                        draft_vocab_size is not None
                        and logits.shape[-1] == draft_vocab_size
                    )

                    if actually_projected:
                        # Already projected inside logits processor
                        all_projected_logits.append(logits.unsqueeze(0))
                        all_target_in_draft_mask.append(
                            timd.unsqueeze(0) if timd is not None else None
                        )
                        any_pre_projected = True
                    elif t2d is not None:
                        # Fallback: logits processor claimed projected but shape disagrees,
                        # or logits_pre_projected was False.  Project here.
                        if logits_pre_projected and logits.shape[-1] != draft_vocab_size:
                            print(
                                f"[Eagle3] WARNING: logits_pre_projected=True but "
                                f"logits.shape[-1]={logits.shape[-1]} != "
                                f"draft_vocab_size={draft_vocab_size}. "
                                f"Applying fallback projection for mb={mb_idx}, sample={i}."
                            )
                        target_max_token = logits.argmax(-1)
                        in_draft_mask = t2d[target_max_token]
                        projected = logits[..., t2d]
                        del logits, target_max_token
                        all_projected_logits.append(projected.unsqueeze(0))
                        all_target_in_draft_mask.append(in_draft_mask.unsqueeze(0))
                        any_pre_projected = True
                    else:
                        all_projected_logits.append(logits.unsqueeze(0))
                        all_target_in_draft_mask.append(None)
                else:
                    all_projected_logits.append(None)
                    all_target_in_draft_mask.append(None)

            # Free memory after each micro-batch
            del data_cache, logits_list, aux_hs_list, timd_list
            torch.cuda.empty_cache()

        # --- Assembly: cat all samples, NO padding here ---
        # Padding is deferred to run_forward() in train_eagle3.py, AFTER
        # get_dp_data_shard_from_tp() reduces the batch from tp_size (e.g. 8)
        # to 1 per GPU.  This avoids the 2x peak memory that padding causes
        # on the large logits tensor (e.g. 9.77 GiB for 8×20480×32000 bf16).

        aux_hidden_states_out = torch.cat(all_aux_hidden_states, dim=0)
        del all_aux_hidden_states

        input_ids_out = torch.cat(all_input_ids, dim=0)
        del all_input_ids

        loss_mask_out = torch.cat(all_loss_mask, dim=0)
        del all_loss_mask

        # Assemble logits: defensive project full-vocab samples, then cat
        t2d = self._t2d
        has_logits = all_projected_logits[0] is not None

        if has_logits:
            draft_vocab_size = int(t2d.sum().item()) if t2d is not None else None
            full_vocab_size = t2d.shape[0] if t2d is not None else None

            for s_idx in range(len(all_projected_logits)):
                logits_s = all_projected_logits[s_idx]
                if logits_s is None:
                    continue
                # Defensive: project full-vocab samples
                if t2d is not None and logits_s.shape[-1] == full_vocab_size:
                    print(
                        f"[Eagle3] WARNING: sample {s_idx} logits has full-vocab "
                        f"shape {logits_s.shape}. Projecting to draft_vocab ({draft_vocab_size})."
                    )
                    logits_2d = logits_s.squeeze(0)  # (S, full_V)
                    max_token = logits_2d.argmax(-1)
                    timd_s = t2d[max_token].unsqueeze(0)
                    logits_s = logits_2d[..., t2d].unsqueeze(0)
                    all_projected_logits[s_idx] = logits_s
                    all_target_in_draft_mask[s_idx] = timd_s
                    any_pre_projected = True
                    del logits_2d, max_token

            target_out = torch.cat(
                [x for x in all_projected_logits if x is not None], dim=0
            )
            del all_projected_logits
            torch.cuda.empty_cache()

            has_timd = any(x is not None for x in all_target_in_draft_mask)
            if has_timd:
                target_in_draft_mask = torch.cat(
                    [x for x in all_target_in_draft_mask if x is not None], dim=0
                )
            else:
                target_in_draft_mask = None
            del all_target_in_draft_mask
        else:
            target_out = None
            target_in_draft_mask = None

        loss_mask_out = loss_mask_out[..., None]

        return Eagle3TargetOutput(
            hidden_states=aux_hidden_states_out,
            target=target_out,
            loss_mask=loss_mask_out,
            input_ids=input_ids_out,
            attention_mask=attention_mask,
            target_in_draft_mask=target_in_draft_mask,
            pre_projected=any_pre_projected,
        )

    def _split_pixel_values_by_grid_thw(
        self,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[List[torch.Tensor]],
        micro_batch_size: int,
    ) -> List[Optional[torch.Tensor]]:
        """Split pixel_values into micro-batches based on image_grid_thw patch counts."""
        if pixel_values is None or image_grid_thw is None:
            return [None]

        if not isinstance(image_grid_thw, (list, tuple)):
            image_grid_thw = [image_grid_thw]

        # Calculate patch counts per sample
        patch_counts = []
        for thw in image_grid_thw:
            if thw is not None:
                if thw.dim() == 1:
                    thw = thw.unsqueeze(0)
                n_patches = (thw[:, 0] * thw[:, 1] * thw[:, 2]).sum().item()
                patch_counts.append(int(n_patches))
            else:
                patch_counts.append(0)

        # Split pixel_values by micro-batch
        splits = []
        offset = 0
        for i in range(0, len(patch_counts), micro_batch_size):
            mb_patch_count = sum(patch_counts[i : i + micro_batch_size])
            if mb_patch_count > 0:
                splits.append(pixel_values[offset : offset + mb_patch_count])
                offset += mb_patch_count
            else:
                splits.append(None)
        return splits

    def _assemble_eagle3_output(
        self,
        data_cache,
        logits_list,
        aux_hidden_states_list,
        last_hidden_states_list,
        attention_mask,
    ) -> Eagle3TargetOutput:
        """Assemble Eagle3TargetOutput from raw extend/extend_vlm results (original path)."""
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

        # --- Defensive fallback: project full-vocab logits ---
        # Check based on actual shape, not flags.  Use chunked projection to
        # avoid OOM during the fallback itself (full-vocab tensor is ~9.5 GiB).
        pre_projected = False
        target_in_draft_mask = None
        if target_out is not None and self._t2d is not None:
            t2d = self._t2d
            full_vocab_size = t2d.shape[0]
            draft_vocab_size = int(t2d.sum().item())
            if target_out.shape[-1] == full_vocab_size:
                print(
                    f"[Eagle3] WARNING: _assemble_eagle3_output received full-vocab "
                    f"logits {target_out.shape}. Applying chunked defensive projection "
                    f"to draft_vocab ({draft_vocab_size})."
                )
                batch_dim = target_out.shape[0]
                seq_dim = target_out.shape[1]
                projected_out = torch.empty(
                    batch_dim, seq_dim, draft_vocab_size,
                    dtype=target_out.dtype, device=target_out.device,
                )
                all_in_draft = torch.empty(
                    batch_dim, seq_dim,
                    dtype=torch.bool, device=target_out.device,
                )
                for b in range(batch_dim):
                    sample_logits = target_out[b]  # (S, V)
                    sample_max = sample_logits.argmax(-1)  # (S,)
                    all_in_draft[b] = t2d[sample_max]
                    projected_out[b] = sample_logits[..., t2d]
                    del sample_logits, sample_max

                del target_out
                torch.cuda.empty_cache()
                target_out = projected_out
                target_in_draft_mask = all_in_draft
                pre_projected = True

        # NOTE: padding is NOT done here — it is deferred to run_forward() in
        # train_eagle3.py, AFTER get_dp_data_shard_from_tp() reduces the batch
        # from tp_size (e.g. 8) to 1.  This avoids 2x peak memory on the large
        # logits tensor (e.g. 9.77 GiB for 8×20480×32000 bf16).
        loss_mask_out = loss_mask_out[..., None]

        return Eagle3TargetOutput(
            hidden_states=aux_hidden_states_out,
            target=target_out,
            loss_mask=loss_mask_out,
            input_ids=input_ids_out,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states_out,
            target_in_draft_mask=target_in_draft_mask,
            pre_projected=pre_projected,
        )

    def _assemble_eagle3_output_projected(
        self,
        data_cache,
        logits_list,
        aux_hidden_states_list,
        target_in_draft_mask_list,
        attention_mask,
    ) -> Eagle3TargetOutput:
        """Assemble Eagle3TargetOutput when logits are pre-projected to draft vocab."""
        aux_hidden_states_out = []
        target_out = []
        loss_mask_out = []
        input_ids_out = []
        timd_out = []

        for data, logits, aux_hs, timd in zip(
            data_cache, logits_list, aux_hidden_states_list, target_in_draft_mask_list
        ):
            aux_hidden_states_out.append(aux_hs.unsqueeze(0))
            loss_mask_out.append(data[2])
            input_ids_out.append(data[0])

            if logits is not None:
                target_out.append(logits.unsqueeze(0))
            else:
                target_out.append(None)

            if timd is not None:
                timd_out.append(timd.unsqueeze(0))
            else:
                timd_out.append(None)

        aux_hidden_states_out = torch.cat(aux_hidden_states_out, dim=0)
        loss_mask_out = torch.cat(loss_mask_out, dim=0)
        input_ids_out = torch.cat(input_ids_out, dim=0)

        if target_out[0] is not None:
            target_out = torch.cat(target_out, dim=0)
        else:
            target_out = None

        has_timd = any(x is not None for x in timd_out)
        if has_timd:
            target_in_draft_mask = torch.cat(
                [x for x in timd_out if x is not None], dim=0
            )
        else:
            target_in_draft_mask = None

        # Safety check: even on the "projected" path, verify the shape
        if target_out is not None and self._t2d is not None:
            t2d = self._t2d
            full_vocab_size = t2d.shape[0]
            draft_vocab_size = int(t2d.sum().item())
            if target_out.shape[-1] == full_vocab_size:
                print(
                    f"[Eagle3] WARNING: _assemble_eagle3_output_projected received "
                    f"full-vocab logits {target_out.shape}! Applying chunked fallback "
                    f"to draft_vocab ({draft_vocab_size})."
                )
                batch_dim = target_out.shape[0]
                seq_dim = target_out.shape[1]
                projected_out = torch.empty(
                    batch_dim, seq_dim, draft_vocab_size,
                    dtype=target_out.dtype, device=target_out.device,
                )
                all_in_draft = torch.empty(
                    batch_dim, seq_dim,
                    dtype=torch.bool, device=target_out.device,
                )
                for b in range(batch_dim):
                    sample_logits = target_out[b]
                    sample_max = sample_logits.argmax(-1)
                    all_in_draft[b] = t2d[sample_max]
                    projected_out[b] = sample_logits[..., t2d]
                    del sample_logits, sample_max
                del target_out
                torch.cuda.empty_cache()
                target_out = projected_out
                target_in_draft_mask = all_in_draft

        # NOTE: padding is NOT done here — it is deferred to run_forward() in
        # train_eagle3.py, AFTER get_dp_data_shard_from_tp() reduces the batch
        # from tp_size (e.g. 8) to 1.  This avoids 2x peak memory on the large
        # logits tensor (e.g. 9.77 GiB for 8×20480×32000 bf16).
        loss_mask_out = loss_mask_out[..., None]

        return Eagle3TargetOutput(
            hidden_states=aux_hidden_states_out,
            target=target_out,
            loss_mask=loss_mask_out,
            input_ids=input_ids_out,
            attention_mask=attention_mask,
            target_in_draft_mask=target_in_draft_mask,
            pre_projected=True,
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
