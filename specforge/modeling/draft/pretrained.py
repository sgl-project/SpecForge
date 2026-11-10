"""Family-independent HF base; legacy base.py remains during migration."""

from itertools import chain

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.modeling_utils import get_torch_context_manager_or_global_device

from .backbone import DraftBackbone
from .hf_config import SpecForgeDraftConfig


class DraftPreTrainedModel(PreTrainedModel):
    config_class = SpecForgeDraftConfig
    _supports_attention_backend = False
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config: SpecForgeDraftConfig):
        if not isinstance(config, SpecForgeDraftConfig):
            raise TypeError("draft models require SpecForgeDraftConfig")
        if type(self).__name__ != config.draft.architecture:
            raise ValueError("model class name must match draft.architecture")
        super().__init__(config)

    def init_weights(self):
        on_meta = [
            tensor.is_meta for tensor in chain(self.parameters(), self.buffers())
        ]
        if (
            on_meta and all(on_meta)
        ) or get_torch_context_manager_or_global_device() == torch.device("meta"):
            self.tie_weights(recompute_mapping=False)
            return
        if any(on_meta):
            raise ValueError("draft is partly on meta; materialize every tensor first")
        super().init_weights()

    def _initialize_weights(self, module, is_custom_code=False):
        # HF's custom-code shortcut skips resets that initialize child tensors.
        super()._initialize_weights(module, False)

    @torch.no_grad()
    def _init_weights(self, module):
        """Own basic initialization without HF family defaults or name matching.

        Components with special parameters or buffers implement reset_draft_parameters().
        They must respect HF's initialized-tensor flags when restoring partial
        weights, for example by using transformers.initialization primitives.
        HF's module flags prevent repeated post_init from overwriting weights.
        """
        custom = getattr(module, "reset_draft_parameters", None)
        if custom is not None:
            custom()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(
                module.weight, mean=0.0, std=self.config.draft.initializer_range
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                if not getattr(module.weight, "_is_hf_initialized", False):
                    init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif any(
            True
            for _ in chain(
                module.parameters(recurse=False), module.buffers(recurse=False)
            )
        ):
            raise TypeError(
                f"{type(module).__name__} must define reset_draft_parameters"
            )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        options = dict(gradient_checkpointing_kwargs or {})
        if options.pop("use_reentrant", False) is not False or options:
            raise ValueError("DraftBackbone supports only use_reentrant=False")
        backbones = [
            module for module in self.modules() if isinstance(module, DraftBackbone)
        ]
        if not backbones:
            raise ValueError("model has no DraftBackbone to checkpoint")
        for backbone in backbones:
            backbone.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if isinstance(module, DraftBackbone):
                module.gradient_checkpointing = False
