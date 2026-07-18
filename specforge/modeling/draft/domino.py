# coding=utf-8
"""Domino draft model entry point."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from specforge.utils import get_device_type

from .dflash import DFlashDraftModel
from .registry import register_draft


@register_draft
class DominoDraftModel(DFlashDraftModel):
    """DFlash backbone with Domino's GRU logits correction."""

    expected_projector_type = "domino"

    def __init__(self, config) -> None:
        dflash_config = getattr(config, "dflash_config", None) or {}
        projector_type = dflash_config.get("projector_type")
        if projector_type is None:
            dflash_config["projector_type"] = self.expected_projector_type
            config.dflash_config = dflash_config
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "DominoDraftModel requires " "dflash_config.projector_type='domino'."
            )
        super().__init__(config)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.emb_dim = int(dflash_config["emb_dim"])
        self.gru_hidden_dim = int(dflash_config["gru_hidden_dim"])
        self.pure_draft_prefix_len = int(dflash_config.get("pure_draft_prefix_len", 0))
        self.shift_label = bool(dflash_config.get("shift_label", False))

        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        in_dim = config.hidden_size + self.gru_hidden_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(in_dim, self.emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.emb_dim, config.vocab_size, bias=False),
        )

        # Ascend NPU DynamicGRU does not support bfloat16. Keep the fp16 copy
        # outside module registration so FSDP does not see mixed parameter dtypes.
        object.__setattr__(self, "_gru_fp16", None)

    @property
    def suffix_start(self) -> int:
        return (
            self.pure_draft_prefix_len
            if self.shift_label
            else (1 + self.pure_draft_prefix_len)
        )

    def _get_gru_fp16(self, device: torch.device) -> nn.GRU:
        gru_fp16 = self._gru_fp16
        if (
            gru_fp16 is None
            or next(gru_fp16.parameters()).device != device
            or gru_fp16.hidden_size != self.prefix_gru.hidden_size
        ):
            gru_fp16 = nn.GRU(
                input_size=self.prefix_gru.weight_ih_l0.shape[1],
                hidden_size=self.prefix_gru.hidden_size,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            gru_fp16.to(device=device, dtype=torch.float16)
            gru_fp16.weight_ih_l0.requires_grad = False
            gru_fp16.weight_hh_l0.requires_grad = False
            object.__setattr__(self, "_gru_fp16", gru_fp16)
        return gru_fp16

    def _run_gru(self, gru_inputs: torch.Tensor) -> torch.Tensor:
        if get_device_type() == "npu" and gru_inputs.dtype == torch.bfloat16:
            gru_fp16 = self._get_gru_fp16(gru_inputs.device)
            gru_fp16.weight_ih_l0.data.copy_(self.prefix_gru.weight_ih_l0.data.half())
            gru_fp16.weight_hh_l0.data.copy_(self.prefix_gru.weight_hh_l0.data.half())
            return gru_fp16(gru_inputs.half())[0].to(gru_inputs.dtype)
        return self.prefix_gru(gru_inputs)[0]

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids
        if prev_token_embeddings is None:
            raise ValueError("DominoDraftModel requires prev_token_embeddings")

        bsz, n_blocks, block_size = base_logits.shape[:3]
        if self.shift_label:
            gru_inputs = prev_token_embeddings.reshape(bsz * n_blocks, block_size, -1)
            gru_out = self._run_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n_blocks, block_size, -1)
            prefix_states = gru_out[:, :, self.suffix_start :, :]
        else:
            gru_inputs = prev_token_embeddings[:, :, : block_size - 1, :].reshape(
                bsz * n_blocks, block_size - 1, -1
            )
            gru_out = self._run_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n_blocks, block_size - 1, -1)
            prefix_states = gru_out[:, :, self.suffix_start - 1 :, :]

        z_n = hidden_states[:, :, self.suffix_start :, :]
        concat_features = torch.cat([z_n, prefix_states], dim=-1)
        logits_e = self.embed_proj(concat_features)

        prefix_logits = base_logits[:, :, : self.suffix_start, :]
        suffix_logits = base_logits[:, :, self.suffix_start :, :] + logits_e
        return torch.cat([prefix_logits, suffix_logits], dim=2)


__all__ = ["DominoDraftModel"]
