"""DSpine owns the decoder/injection loop and preserves intermediate states."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import initialization as init
from transformers.utils import ModelOutput

from ..heads.adjacent_injection import AdjacentInjection
from ..heads.transfer_space import initialize_transfer_space
from ..hf_config import SpecForgeDraftConfig
from ..layers.inputs import LayerInputs
from ..pretrained import DraftPreTrainedModel
from .assembly import build_target_context_modules
from .context import prepare_context_inputs
from .registry import register_draft_model


@dataclass
class DSpineOutput(ModelOutput):
    """States/logits retain block axes; layer_features adds a leading layer axis."""

    hidden_states: Tensor | None = None
    pre_injection: Tensor | None = None
    layer_features: Tensor | None = None
    logits: Tensor | None = None


@register_draft_model
class DSpineDraftModel(DraftPreTrainedModel):
    """Consume explicit block visibility, token blocks and per-layer replacement masks.

    The caller prepares replacement masks; training schedules and layer cutoffs
    do not enter this model. Transfer initialization reads the bound target head
    and is called only when starting a new transfer space.
    """

    _no_split_modules = ["DecoderLayer"]

    def __init__(
        self, config: SpecForgeDraftConfig, *, backend="eager", device=None, dtype=None
    ):
        super().__init__(config)
        draft, spec = config.draft, config.draft.spec_config
        self.backbone, self.fusion, self.embed_tokens, self.lm_head = (
            build_target_context_modules(
                draft, spec.target_features, backend=backend, device=device, dtype=dtype
            )
        )
        self.injection = AdjacentInjection(
            spec.injection,
            hidden_size=draft.hidden_size,
            num_layers=draft.num_hidden_layers,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "transfer_codes",
            torch.zeros(
                draft.vocab_size,
                spec.injection.transfer_rank,
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer("transfer_ready", torch.tensor(False, device=device))
        self.post_init()

    def reset_draft_parameters(self):
        init.zeros_(self.transfer_codes)
        init.zeros_(self.transfer_ready)

    def finalize_initialization(self, *, loaded_keys: frozenset[str]) -> None:
        transfer_keys = {"transfer_codes", "transfer_ready"} | {
            f"injection.readouts.{name}"
            for name in self.injection.readouts.state_dict()
        }
        loaded = transfer_keys & loaded_keys
        if loaded and (loaded != transfer_keys or not self.transfer_ready.item()):
            raise ValueError(
                "Cannot initialize an incomplete DSpine transfer space over loaded state; "
                "load the complete transfer space or mark all of it allow_new"
            )
        if not loaded:
            self.initialize_transfer_space()

    @torch.no_grad()
    def initialize_transfer_space(self, *, chunk_size: int = 4096) -> None:
        if self.transfer_ready.item():
            raise RuntimeError("DSpine transfer space is already initialized")
        initialize_transfer_space(
            self.lm_head.bound_weight(),
            self.transfer_codes,
            self.injection.readouts,
            chunk_size=chunk_size,
        )
        self.transfer_ready.fill_(True)

    def _require_transfer_space(self):
        torch._assert_async(
            self.transfer_ready,
            "Initialize or load the DSpine transfer space before use",
        )

    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        layer_inputs: Sequence[LayerInputs],
        *,
        block_token_ids: Tensor,
        replace_masks: Sequence[Tensor | None] | None = None,
        output_logits: bool = True,
    ) -> DSpineOutput:
        self._require_transfer_space()
        size = self.config.draft.spec_config.protocol.num_model_slots
        if input_ids.ndim != 2 or input_ids.shape[1] % size:
            raise ValueError("input_ids must be [batch, whole draft blocks]")
        shape = (input_ids.shape[0], input_ids.shape[1] // size, size)
        if block_token_ids.shape != shape or block_token_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "block_token_ids must be integer [batch, blocks, block_size]"
            )
        if type(output_logits) is not bool:
            raise TypeError("output_logits must be boolean")
        masks = (
            (None,) * self.backbone.num_layers
            if replace_masks is None
            else tuple(replace_masks)
        )
        if len(masks) != self.backbone.num_layers:
            raise ValueError("replace_masks must contain one entry per layer")
        if any(
            mask is not None and (mask.shape != shape[:2] or mask.dtype != torch.bool)
            for mask in masks
        ):
            raise ValueError("replacement masks must be boolean [batch, blocks]")
        prepared = prepare_context_inputs(
            self.fusion,
            target_features,
            layer_inputs,
            num_layers=self.backbone.num_layers,
        )
        state = self.backbone.begin(self.embed_tokens(input_ids), prepared)
        features = []
        for index, replace_mask in enumerate(masks):
            state = self.backbone.run_next_layer(state)
            pre_injection = state.hidden.reshape(*shape, state.hidden.shape[-1])
            hidden = self.injection(
                pre_injection, index, block_token_ids, self.transfer_codes, replace_mask
            )
            features.append(self.injection.feature(hidden, index))
            state = state.with_hidden(hidden.reshape_as(state.hidden))
        hidden = self.backbone.finalize(state).reshape_as(pre_injection)
        return DSpineOutput(
            hidden_states=hidden,
            pre_injection=pre_injection,
            layer_features=torch.stack(features),
            logits=self.compute_logits(hidden) if output_logits else None,
        )

    def refine(self, hidden: Tensor, predecessor_ids: Tensor) -> Tensor:
        """Reuse the final injection and norm on pre-injection successor states."""
        self._require_transfer_space()
        if (
            hidden.shape[-1] != self.config.draft.hidden_size
            or predecessor_ids.shape != hidden.shape[:-1]
            or predecessor_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("predecessor_ids must align with hidden states")
        layer = self.backbone.num_layers - 1
        receiver = self.injection.feature(hidden, layer)
        predecessor = self.injection.code(predecessor_ids, self.transfer_codes)
        return self.backbone.final_norm(
            self.injection.write(hidden, receiver, predecessor, layer)
        )
