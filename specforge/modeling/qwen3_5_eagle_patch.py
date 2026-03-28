"""
Monkey patch to add EAGLE3 support to Qwen3.5 model in sglang.
This file patches both the set_eagle3_layers_to_capture method and the forward method
to properly capture hidden states for EAGLE3 training.

Environment Variables:
- QWEN35_EAGLE3_ENABLE: Enable/disable EAGLE3 patch (default: auto-detect)
- QWEN35_EAGLE3_DEBUG: Enable debug logging (default: false)
"""

import os
import torch
import sys
from typing import List, Optional, Union, Dict

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

# Store original functions
_original_general_mm_embed_routine = None
_original_vlm_forward = None


def _patched_vlm_forward(self, input_ids, positions, forward_batch, pp_proxy_tensors=None, get_embedding=False):
    """
    Patched forward method for Qwen3_5ForConditionalGeneration that passes aux_hidden_states to logits processor.
    """
    if self.is_mrope_enabled:
        positions = forward_batch.mrope_positions

    if not (
        forward_batch.forward_mode.is_decode()
        or not forward_batch.contains_image_inputs()
    ):
        if self.is_mrope_enabled:
            assert positions.ndim == 2 and positions.size(0) == 3, (
                "multimodal section rotary embedding requires "
                f"(3, seq_len) positions, but got {positions.size()}"
            )

    # Call the patched general_mm_embed_routine which stores aux_hidden_states in the language model
    from sglang.srt.managers.mm_utils import general_mm_embed_routine
    hidden_states = general_mm_embed_routine(
        input_ids=input_ids,
        forward_batch=forward_batch,
        language_model=self.model,
        multimodal_model=self,
        positions=positions,
        use_deepstack=self.use_deepstack,
        pp_proxy_tensors=pp_proxy_tensors,
    )

    if self.pp_group.is_last_rank:
        if not get_embedding:
            # Retrieve aux_hidden_states from language model if available
            aux_hidden_states = getattr(self.model, '_eagle3_aux_hidden_states', None)
            # Clear the stored value to avoid memory leak
            if hasattr(self.model, '_eagle3_aux_hidden_states'):
                delattr(self.model, '_eagle3_aux_hidden_states')
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states=aux_hidden_states,
            )
        else:
            return self.pooler(hidden_states, forward_batch)
    else:
        return hidden_states


def _patched_general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: torch.nn.Module,
    multimodal_model: Optional[torch.nn.Module] = None,
    data_embedding_funcs: Dict = None,
    placeholder_tokens: Optional[dict] = None,
    use_deepstack: Dict = {},
    **kwargs,
) -> torch.Tensor:
    """
    Patched version of general_mm_embed_routine that handles (hidden_states, aux_hidden_states) tuple.
    Stores aux_hidden_states in the model for later retrieval by logits processor.
    """
    global _original_general_mm_embed_routine

    # Call original function
    result = _original_general_mm_embed_routine(
        input_ids=input_ids,
        forward_batch=forward_batch,
        language_model=language_model,
        multimodal_model=multimodal_model,
        data_embedding_funcs=data_embedding_funcs,
        placeholder_tokens=placeholder_tokens,
        use_deepstack=use_deepstack,
        **kwargs,
    )

    # Handle tuple return from patched language model forward
    if isinstance(result, tuple):
        hidden_states, aux_hidden_states = result
        # Store aux_hidden_states in the language model for later access
        language_model._eagle3_aux_hidden_states = aux_hidden_states
        return hidden_states
    return result

# 缓存已补丁的实例
_patched_instances = set()
_original_forwards = {}


def _qwen3_5_forward_with_eagle3(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: Optional[torch.Tensor] = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    input_deepstack_embeds: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple, PPProxyTensors]:
    """
    Modified forward method for Qwen3_5ForCausalLM that captures aux hidden states for EAGLE3.
    Based on the original forward method, with aux_hidden_states capture logic added.
    """
    # Initialize hidden states
    if self.pp_group.is_first_rank:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
    else:
        assert pp_proxy_tensors is not None
        hidden_states = pp_proxy_tensors["hidden_states"]
        residual = pp_proxy_tensors["residual"]

    # Capture aux hidden states for EAGLE3
    aux_hidden_states = []
    should_capture = (
        hasattr(self, 'capture_aux_hidden_states')
        and self.capture_aux_hidden_states
        and hasattr(self, 'layers_to_capture')
        and self.pp_group.is_last_rank
    )

    # Pass through decoder layers
    for layer_idx in range(len(self.layers)):
        # Capture hidden states before processing this layer if needed
        if should_capture and layer_idx in self.layers_to_capture:
            if residual is None:
                # First layer doesn't have residual yet
                aux_hidden_states.append(hidden_states.clone())
            else:
                aux_hidden_states.append(hidden_states + residual)

        layer = self.layers[layer_idx]

        # Try to use expert recorder if available, otherwise skip
        try:
            from sglang.srt.layers.moe.fused_moe_triton import get_global_expert_distribution_recorder
            with get_global_expert_distribution_recorder().with_current_layer(layer_idx):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                )
        except (ImportError, AttributeError):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        # Process deepstack embeddings if provided
        if (
            input_deepstack_embeds is not None
            and input_deepstack_embeds.numel() > 0
            and layer_idx < 3
        ):
            sep = self.hidden_size * layer_idx
            hidden_states.add_(
                input_deepstack_embeds[:, sep : sep + self.hidden_size]
            )

    # Return intermediate tensors for pipeline parallelism
    if not self.pp_group.is_last_rank:
        return PPProxyTensors(
            {
                "hidden_states": hidden_states,
                "residual": residual,
            }
        )

    # Apply final normalization
    if hidden_states.shape[0] != 0:
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

    # Return with aux_hidden_states if captured
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


def patch_qwen3_5_instance(model_instance):
    """
    Patch an individual model instance for EAGLE3 training.
    Works for both Qwen3_5ForCausalLM and Qwen3_5ForConditionalGeneration.
    """

    # 避免重复补丁
    instance_id = id(model_instance)
    if instance_id in _patched_instances:
        return

    try:
        # 确定这是 VLM wrapper 还是直接的语言模型
        if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'layers'):
            # VLM wrapper: Qwen3_5ForConditionalGeneration
            lm_model = model_instance.model
            is_vlm = True
        elif hasattr(model_instance, 'layers'):
            # 直接的语言模型: Qwen3_5ForCausalLM
            lm_model = model_instance
            is_vlm = False
        else:
            print(f"Warning: Unknown model structure: {type(model_instance)}")
            return

        # 添加必要的属性到 wrapper (如果是 VLM)
        if is_vlm:
            if not hasattr(model_instance, 'capture_aux_hidden_states'):
                model_instance.capture_aux_hidden_states = False
            if not hasattr(model_instance, 'layers_to_capture'):
                model_instance.layers_to_capture = []
            if not hasattr(model_instance, 'aux_hidden_states'):
                model_instance.aux_hidden_states = []

        # 添加必要的属性到语言模型
        if not hasattr(lm_model, 'capture_aux_hidden_states'):
            lm_model.capture_aux_hidden_states = False
        if not hasattr(lm_model, 'layers_to_capture'):
            lm_model.layers_to_capture = []
        if not hasattr(lm_model, 'aux_hidden_states'):
            lm_model.aux_hidden_states = []

        # 定义 set_eagle3_layers_to_capture 方法
        def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
            # Find the actual language model
            if hasattr(self, 'model') and hasattr(self.model, 'layers'):
                lm_model = self.model
                lm_config = self.model.config
            elif hasattr(self, 'layers'):
                lm_model = self
                lm_config = self.config
            else:
                raise AttributeError(f"Cannot find layers in {type(self)}")

            if not hasattr(self, 'pp_group') or not self.pp_group.is_last_rank:
                return

            self.capture_aux_hidden_states = True
            lm_model.capture_aux_hidden_states = True

            if layer_ids is None:
                num_layers = lm_config.num_hidden_layers
                layer_ids = [2, num_layers // 2, num_layers - 3]

            # In sglang, we need to add 1 to layer_ids
            self.layers_to_capture = [val + 1 for val in layer_ids]
            lm_model.layers_to_capture = [val + 1 for val in layer_ids]

            self.aux_hidden_states = [None] * len(layer_ids)
            lm_model.aux_hidden_states = [None] * len(layer_ids)

            # Mark layers to capture
            for layer_idx, layer in enumerate(lm_model.layers):
                layer.should_capture_aux_hidden_state = layer_idx in lm_model.layers_to_capture

        # 绑定方法到实例
        import types
        model_instance.set_eagle3_layers_to_capture = types.MethodType(
            set_eagle3_layers_to_capture, model_instance
        )

        # 只修补语言模型的 forward 方法
        if id(lm_model) not in _original_forwards:
            original_forward = lm_model.forward
            _original_forwards[id(lm_model)] = original_forward
            lm_model.forward = types.MethodType(
                _qwen3_5_forward_with_eagle3, lm_model
            )

        _patched_instances.add(instance_id)

    except Exception as e:
        print(f"Error patching model instance: {e}")
        import traceback
        traceback.print_exc()


def patch_qwen3_5_for_eagle3():
    """Patch Qwen3.5 model class for EAGLE3 training."""

    global _original_general_mm_embed_routine

    try:
        # Patch general_mm_embed_routine to handle tuple returns
        if _original_general_mm_embed_routine is None:
            try:
                from sglang.srt.managers.mm_utils import general_mm_embed_routine
                _original_general_mm_embed_routine = general_mm_embed_routine
                import sglang.srt.managers.mm_utils as mm_utils_module
                mm_utils_module.general_mm_embed_routine = _patched_general_mm_embed_routine
                print("Successfully patched general_mm_embed_routine")
            except Exception as e:
                print(f"Warning: Could not patch general_mm_embed_routine: {e}")

        # 尝试导入并补丁类
        if 'sglang.srt.models.qwen3_5' in sys.modules:
            # 如果已导入，删除缓存以便重新加载
            del sys.modules['sglang.srt.models.qwen3_5']
            # 同时删除相关模块
            for mod in list(sys.modules.keys()):
                if 'sglang.srt.models.qwen3_5' in mod:
                    del sys.modules[mod]

        from sglang.srt.models.qwen3_5 import (
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
        )

        # 定义通用的 set_eagle3_layers_to_capture 方法
        def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
            # Find the actual language model
            if hasattr(self, 'model') and hasattr(self.model, 'layers'):
                lm_model = self.model
                lm_config = self.model.config
            elif hasattr(self, 'layers'):
                lm_model = self
                lm_config = self.config
            else:
                raise AttributeError(f"Cannot find layers in {type(self)}")

            if not hasattr(self, 'pp_group') or not self.pp_group.is_last_rank:
                return

            self.capture_aux_hidden_states = True
            lm_model.capture_aux_hidden_states = True

            if layer_ids is None:
                num_layers = lm_config.num_hidden_layers
                layer_ids = [2, num_layers // 2, num_layers - 3]

            self.layers_to_capture = [val + 1 for val in layer_ids]
            lm_model.layers_to_capture = [val + 1 for val in layer_ids]

            self.aux_hidden_states = [None] * len(layer_ids)
            lm_model.aux_hidden_states = [None] * len(layer_ids)

            for layer_idx, layer in enumerate(lm_model.layers):
                layer.should_capture_aux_hidden_state = layer_idx in lm_model.layers_to_capture

        # 补丁 Qwen3_5ForCausalLM - 替换 forward 方法
        if not hasattr(Qwen3_5ForCausalLM, '_eagle3_class_patched'):
            # 保存原始 forward
            original_forward = Qwen3_5ForCausalLM.forward
            Qwen3_5ForCausalLM._original_forward = original_forward
            Qwen3_5ForCausalLM._eagle3_class_patched = True

            # 添加 set_eagle3_layers_to_capture 方法
            Qwen3_5ForCausalLM.set_eagle3_layers_to_capture = set_eagle3_layers_to_capture

            # 替换 forward 方法
            Qwen3_5ForCausalLM.forward = _qwen3_5_forward_with_eagle3

            print("Successfully patched Qwen3_5ForCausalLM class")

        # 补丁 Qwen3_5ForConditionalGeneration - 添加方法并替换 forward
        if not hasattr(Qwen3_5ForConditionalGeneration, '_eagle3_class_patched'):
            # 保存原始 forward
            original_vlm_forward = Qwen3_5ForConditionalGeneration.forward
            Qwen3_5ForConditionalGeneration._original_forward = original_vlm_forward
            Qwen3_5ForConditionalGeneration._eagle3_class_patched = True

            # 添加 set_eagle3_layers_to_capture 方法
            Qwen3_5ForConditionalGeneration.set_eagle3_layers_to_capture = set_eagle3_layers_to_capture

            # 替换 forward 方法
            Qwen3_5ForConditionalGeneration.forward = _patched_vlm_forward

            print("Successfully patched Qwen3_5ForConditionalGeneration class")

    except Exception as e:
        print(f"Warning: Could not patch Qwen3_5 class: {e}")
        import traceback
        traceback.print_exc()


# 初始化类补丁（通过环境变量控制）
def _should_enable_patch():
    """Check if EAGLE3 patch should be enabled."""
    env_value = os.getenv('QWEN35_EAGLE3_ENABLE', '').lower()
    if env_value in ('1', 'true', 'yes', 'on'):
        return True
    if env_value in ('0', 'false', 'no', 'off'):
        return False
    # Auto-detect: check if running EAGLE3 training
    return '--eagle' in sys.argv or 'eagle' in ' '.join(sys.argv).lower()


# 只在需要时初始化类补丁
if _should_enable_patch():
    patch_qwen3_5_for_eagle3()
elif os.getenv('QWEN35_EAGLE3_DEBUG', '').lower() in ('1', 'true'):
    print("[Qwen35Eagle3] Patch disabled (QWEN35_EAGLE3_ENABLE not set)")


# 导出控制函数供外部使用
def enable_qwen35_eagle3_patch():
    """Manually enable Qwen3.5 EAGLE3 patch."""
    patch_qwen3_5_for_eagle3()


def is_qwen35_eagle3_enabled():
    """Check if Qwen3.5 EAGLE3 patch is enabled."""
    return _should_enable_patch()
