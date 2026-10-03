"""DFlash-family model adapter for the TorchTitan training engine.

The algorithm modules remain SpecForge modules.  This is deliberately a
``partial_dtensor`` adapter, not a claim that Hugging Face modules implement
TorchTitan's native ``spmd_types`` module protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torchtitan.protocols.model import BaseModel
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDSparkModel,
)
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel


class _DecoderLayers(nn.ModuleList):
    """Keep SpecForge's list iteration and TorchTitan's named-block interface."""

    def items(self):
        return self._modules.items()


def _load_weights(path: str) -> dict[str, torch.Tensor]:
    source = Path(path)
    if source.is_dir():
        for index_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            index = source / index_name
            if index.exists():
                weight_map = json.loads(index.read_text())["weight_map"]
                result = {}
                for shard in sorted(set(weight_map.values())):
                    result.update(_load_weights(str(source / shard)))
                return result
        for filename in ("model.safetensors", "pytorch_model.bin"):
            if (source / filename).exists():
                return _load_weights(str(source / filename))
        raise FileNotFoundError(f"No model weights found in {source}")
    if source.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(source), device="cpu")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(f"Expected a tensor state dict in {source}")
    return state


def _build_training_model(config: "SpecForgeTitanModel.Config") -> nn.Module:
    draft_config = Qwen3Config.from_dict(config.draft_config)
    attention = config.objective.get("attention_backend", "sdpa")
    draft_config._attn_implementation = attention
    draft_classes = {
        "dflash": DFlashDraftModel,
        "dflash2": DFlash2DraftModel,
        "dspark": DSparkDraftModel,
    }
    if config.algorithm not in draft_classes:
        raise ValueError(f"Unknown DFlash-family algorithm: {config.algorithm}")
    draft = draft_classes[config.algorithm](draft_config)
    draft.layers = _DecoderLayers(draft.layers)
    # These are real adapter attributes, not replacement registered modules.
    # The trainable decoder has no token embedding or vocabulary projection.
    draft.enable_weight_tying = False
    draft.tok_embeddings = None
    draft.lm_head = None
    embedding = nn.Embedding(
        draft_config.vocab_size,
        draft_config.hidden_size,
        dtype=getattr(torch, config.teacher_dtype),
    )
    head = nn.Linear(
        draft_config.hidden_size,
        draft_config.vocab_size,
        bias=False,
        dtype=getattr(torch, config.teacher_dtype),
    )
    embedding.requires_grad_(False)
    head.requires_grad_(False)
    objective = dict(config.objective)
    objective.setdefault("block_size", draft.block_size)
    objective.setdefault("mask_token_id", draft.mask_token_id)
    objective.setdefault("attention_backend", attention)
    if objective["mask_token_id"] is None:
        raise ValueError("DFlash training requires a mask_token_id")
    training_cls = (
        OnlineDSparkModel if config.algorithm == "dspark" else OnlineDFlashModel
    )
    return training_cls(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embedding,
        **objective,
    )


class SpecForgeTitanModel(BaseModel):
    """A TorchTitan model whose forward computes SpecForge's draft objective."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        draft_config: dict[str, Any]
        algorithm: str = "dflash"
        objective: dict[str, Any] = field(default_factory=dict)
        teacher_state_path: str | None = None
        draft_state_path: str | None = None
        init_seed: int = 42
        teacher_dtype: str = "bfloat16"

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            self.teacher_dtype = config.training.mixed_precision_param
            if config.parallelism.spmd_backend != "partial_dtensor":
                raise ValueError(
                    "SpecForge's TorchTitan model adapter currently requires "
                    "parallelism.spmd_backend='partial_dtensor'."
                )

        def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, int]:
            total = sum(p.numel() for p in model.parameters())
            draft = model.draft_model
            anchors = min(model.training_model.num_anchors, max(seq_len - 1, 1))
            queries = anchors * draft.block_size
            # Analytical forward+backward estimate per teacher input token.
            # As in Titan's decoder estimates, this assumes dense attention
            # and all configured anchors are valid. Sparse masks, padded
            # examples, auxiliary elementwise work and checkpoint recompute
            # make MFU a diagnostic estimate, not a measured FLOP count.
            flops = 0
            for name, module in draft.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                if name == "fc":
                    length = seq_len
                elif name.endswith(("self_attn.k_proj", "self_attn.v_proj")):
                    length = seq_len + queries
                else:
                    length = queries
                flops += 6 * length * module.weight.numel()
            # The frozen vocabulary projection needs forward and input
            # gradients, but has no weight-gradient GEMM.
            flops += 4 * queries * model.training_model.lm_head.weight.numel()
            head_dim = getattr(
                draft.config,
                "head_dim",
                draft.config.hidden_size // draft.config.num_attention_heads,
            )
            flops += (
                12
                * len(draft.layers)
                * queries
                * (seq_len + queries)
                * draft.config.num_attention_heads
                * head_dim
            )
            return total, max(flops // seq_len, 1)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.training_model = _build_training_model(config)

    @property
    def draft_model(self):
        return self.training_model.draft_model

    def verify_module_protocol(self) -> None:
        # BaseModel explicitly allows adapters containing external nn.Modules
        # to override this check. Their parallelization and initialization are
        # supplied below rather than Module.parallelize()/init_states().
        if not isinstance(self.training_model, (OnlineDFlashModel, OnlineDSparkModel)):
            raise TypeError("Expected a SpecForge DFlash-family training model")

    def init_weights(self, **kwargs) -> None:
        del kwargs
        # Initialize a full reference, then let DCP slice it into the actual
        # DP/TP placements. Initializing local TP shards independently changes
        # the seeded model when the parallelism degree changes.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(self.config.init_seed)
            reference = _build_training_model(self.config)
            if self.config.draft_state_path:
                reference.draft_model.load_state_dict(
                    _load_weights(self.config.draft_state_path), strict=True
                )
            if self.config.teacher_state_path:
                teacher = _load_weights(self.config.teacher_state_path)
                for name, module in (
                    ("lm_head", reference.lm_head),
                    ("embed_tokens", reference.embed_tokens),
                ):
                    key = f"{name}.weight"
                    if key not in teacher:
                        raise ValueError(f"Teacher state is missing {key}")
                    module.load_state_dict({"weight": teacher[key]}, strict=True)
            state = {
                f"training_model.{name}": value
                for name, value in reference.state_dict().items()
            }
            # Pipeline model parts retain their original global layer names
            # but own only a subset of the full model's state.
            live_keys = set(self.state_dict())
            state = {name: value for name, value in state.items() if name in live_keys}
            set_model_state_dict(
                self,
                state,
                options=StateDictOptions(full_state_dict=True, strict=True),
            )
            # HF RoPE uses nonpersistent buffers, which DCP intentionally does
            # not include in state_dict. to_empty() discarded their contents.
            reference_buffers = dict(reference.named_buffers())
            for name, buffer in self.training_model.named_buffers():
                if name in reference_buffers:
                    buffer.copy_(reference_buffers[name].to(buffer.device))

    def forward(self, input_ids: torch.Tensor, **kwargs):
        normalizer = kwargs.pop("objective_normalizer", None)
        loss, accuracy, metrics = self.training_model(input_ids=input_ids, **kwargs)
        if normalizer is not None:
            metrics = dict(metrics)
            metrics["_torchtitan_normalizer"] = normalizer
        return loss, accuracy, metrics
