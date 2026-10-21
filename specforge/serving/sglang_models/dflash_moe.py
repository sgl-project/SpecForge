# coding=utf-8
"""SGLang draft classes for DFlash-family drafters with the ``qwen3_5_moe`` FFN.

``Qwen3MoeDSparkModel`` and ``DFlashMoEDraftModel`` are SGLang's
``Qwen3DSparkModel`` / ``DFlashDraftModel`` with every decoder layer's dense
MLP replaced by the modules SGLang's Qwen MoE targets run: a ``TopK`` router
step, a ``FusedMoE`` holding the routed experts and a sigmoid-gated shared
expert. Expert weights load through ``FusedMoE``'s own loader, so kernel
layouts, quantisation and tensor-parallel sharding stay SGLang's business.

The recipe comes from the export's ``config.json`` as SpecForge writes it:
``num_experts``, ``num_experts_per_tok``, ``moe_intermediate_size``,
``shared_expert_intermediate_size`` (0 = no shared expert), ``norm_topk_prob``
and ``moe_router_bias`` (router centering folded into ``gate.bias`` at export).
The export's experts are stacked (``experts.gate_up_proj`` [E, 2I, H],
``experts.down_proj`` [E, H, I]); per-expert ``experts.{i}.gate_proj`` names
are accepted too. Router logits are computed in fp32 as in training.

Registered through ``SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models``.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Set, Tuple

import torch
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.models.dflash import DFlashDecoderLayer, DFlashDraftModel
from sglang.srt.models.dspark import Qwen3DSparkModel
from torch import nn

logger = logging.getLogger(__name__)

_EXPERT_TENSOR = re.compile(
    r"^experts\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$"
)
_LAYER_FFN = re.compile(r"^(?:model\.)?layers\.(?P<idx>\d+)\.mlp\.(?P<key>.+)$")
# Qwen names, and the DeepSeek-style w1 (gate) / w2 (down) / w3 (up) some exports use.
_SHARD = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
_SHARD.update({shard: shard for shard in ("w1", "w2", "w3")})


def _cfg(config, key: str, default):
    value = getattr(config, key, None)
    return default if value is None else value


def num_routed_experts(config) -> int:
    """``num_experts`` of a draft config; 0 means a dense draft."""
    return int(_cfg(config, "num_experts", 0) or 0)


class QwenSharedExpertMLP(nn.Module):
    """The shared expert as SGLang builds Qwen MoE MLPs: merged gate/up, SiluAndMul, down."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, quant_config, prefix: str
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gate_up_proj(x)
        y, _ = self.down_proj(self.act_fn(h))
        return y


class QwenMoESparseBlock(nn.Module):
    """``FusedMoE(x, TopK(gate(x))) + sigmoid(shared_expert_gate(x)) * shared_expert(x)``."""

    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = num_routed_experts(config)
        self.top_k = int(_cfg(config, "num_experts_per_tok", 0))
        self.intermediate_size = int(_cfg(config, "moe_intermediate_size", 0))
        if min(self.num_experts, self.top_k, self.intermediate_size) <= 0:
            raise ValueError(
                "MoE draft config needs num_experts, num_experts_per_tok and "
                f"moe_intermediate_size; got {self.num_experts}, {self.top_k}, "
                f"{self.intermediate_size}"
            )
        if str(_cfg(config, "hidden_act", "silu")) != "silu":
            raise ValueError("MoE drafts serve silu experts only")
        self.norm_topk_prob = bool(_cfg(config, "norm_topk_prob", True))
        self.router_bias = bool(_cfg(config, "moe_router_bias", False))
        shared_size = int(_cfg(config, "shared_expert_intermediate_size", 0) or 0)

        # The router runs in fp32 like the trainer; ReplicatedLinear for its loader.
        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_experts,
            bias=self.router_bias,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.topk = TopK(
            top_k=self.top_k, renormalize=self.norm_topk_prob, layer_id=layer_id
        )
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            layer_id=layer_id,
            top_k=self.top_k,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            inplace=False,
        )
        self.shared_expert: Optional[QwenSharedExpertMLP] = None
        self.shared_expert_gate: Optional[nn.Linear] = None
        if shared_size > 0:
            self.shared_expert = QwenSharedExpertMLP(
                self.hidden_size, shared_size, quant_config, f"{prefix}.shared_expert"
            )
            self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        self._loaded_experts: Set[Tuple[int, str]] = set()
        self._loaded_params: Set[str] = set()

    def describe(self) -> str:
        shared = "none"
        if self.shared_expert is not None:
            shared = f"{self.shared_expert.down_proj.input_size} (sigmoid-gated)"
        return (
            f"{self.num_experts} routed experts, top-{self.top_k}, width "
            f"{self.intermediate_size}, shared={shared}, router_bias={self.router_bias}, "
            f"renorm={self.norm_topk_prob}, runner={type(self.experts.quant_method).__name__}"
        )

    def router_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.gate(x.float())
        return logits

    def selected_experts(self, x: torch.Tensor) -> torch.Tensor:
        """Expert ids the router picks, ``[tokens, top_k]``; compared by the gate."""
        return self.router_logits(x).topk(self.top_k, dim=-1).indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        y = self.experts(x, self.topk(x, self.router_logits(x)))
        if isinstance(y, tuple):  # some SGLang builds return (out, bias)
            y = y[0]
        if self.shared_expert is not None:
            shared = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                gate = torch.sigmoid(self.shared_expert_gate(x).float())
                shared = gate.to(shared.dtype) * shared
            y = y + shared
        return y.view(shape)

    # -- loading ---------------------------------------------------------

    def _load_expert(self, expert_id: int, shard_id: str, tensor: torch.Tensor):
        name = "experts.w2_weight" if shard_id == "w2" else "experts.w13_weight"
        param = dict(self.named_parameters())[name]
        param.weight_loader(param, tensor, name, shard_id=shard_id, expert_id=expert_id)
        self._loaded_experts.add((expert_id, shard_id))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load this block's entries (names relative to the block) via SGLang's loaders."""
        params = dict(self.named_parameters())
        for name, tensor in weights:
            if name == "experts.gate_up_proj":  # stacked [E, 2I, H]: gate rows, then up
                for expert_id, gate_up in enumerate(tensor):
                    gate, up = gate_up.chunk(2, dim=0)
                    self._load_expert(expert_id, "w1", gate)
                    self._load_expert(expert_id, "w3", up)
                continue
            if name == "experts.down_proj":  # stacked [E, H, I]
                for expert_id, down in enumerate(tensor):
                    self._load_expert(expert_id, "w2", down)
                continue
            match = _EXPERT_TENSOR.match(name)
            if match is not None:
                self._load_expert(int(match["idx"]), _SHARD[match["proj"]], tensor)
                continue
            if name.startswith(("shared_expert.gate_proj.", "shared_expert.up_proj.")):
                param = params["shared_expert.gate_up_proj.weight"]
                param.weight_loader(param, tensor, 0 if "gate_proj" in name else 1)
                self._loaded_params.add(name)
                continue
            if name not in params:
                raise ValueError(
                    f"checkpoint FFN entry {name!r} does not map to any parameter of "
                    f"{type(self).__name__}; does config.json's architectures match the export?"
                )
            param = params[name]
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, tensor)
            else:
                with torch.no_grad():
                    param.copy_(tensor.to(param.dtype))
            self._loaded_params.add(name)

    def check_loaded(self, where: str) -> None:
        """Refuse to serve a partially loaded FFN."""
        missing: List[str] = [
            f"experts.{expert_id}.{shard}"
            for expert_id in range(self.num_experts)
            for shard in ("w1", "w2", "w3")
            if (expert_id, shard) not in self._loaded_experts
        ]
        expected = {"gate.weight"} | ({"gate.bias"} if self.router_bias else set())
        if self.shared_expert is not None:
            expected |= {
                "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight",
                "shared_expert.down_proj.weight",
                "shared_expert_gate.weight",
            }
        missing += sorted(expected - self._loaded_params)
        if missing:
            raise ValueError(
                f"{where}: {len(missing)} FFN tensor(s) missing from the checkpoint, "
                f"would serve uninitialised: {', '.join(missing[:8])}"
            )


class QwenMoEDecoderLayer(DFlashDecoderLayer):
    """``DFlashDecoderLayer`` with the dense MLP replaced by :class:`QwenMoESparseBlock`."""

    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__(config, layer_id, quant_config)
        # Drop the dense MLP the base layer built; the export has no weights for it.
        del self.mlp
        self.mlp = QwenMoESparseBlock(
            config, layer_id, quant_config, prefix=f"layers.{layer_id}.mlp"
        )


class _QwenMoEDraftMixin:
    """MoE decoder layers, strict FFN loading, one log line per model."""

    decoder_layer_cls = QwenMoEDecoderLayer

    def __init__(self, config, *args, **kwargs) -> None:
        if num_routed_experts(config) <= 0:
            raise ValueError(
                f"{type(self).__name__} needs num_experts > 0; use the dense class"
            )
        super().__init__(config, *args, **kwargs)
        logger.info(
            "MoE draft (%s): %s", type(self).__name__, self.layers[0].mlp.describe()
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Every layers.{i}.mlp.* tensor goes to that layer's block; the base loader takes the rest.
        rest: List[Tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            match = _LAYER_FFN.match(name)
            if match is None:
                rest.append((name, tensor))
                continue
            idx = int(match["idx"])
            if idx >= len(self.layers):
                raise ValueError(
                    f"checkpoint has FFN weights for layer {idx}; the draft has {len(self.layers)}"
                )
            self.layers[idx].mlp.load_weights([(match["key"], tensor)])
        super().load_weights(iter(rest))
        for idx, layer in enumerate(self.layers):
            layer.mlp.check_loaded(f"{type(self).__name__} layers.{idx}.mlp")


class DFlashMoEDraftModel(_QwenMoEDraftMixin, DFlashDraftModel):
    """DFlash draft with the ``qwen3_5_moe`` FFN."""


class Qwen3MoeDSparkModel(_QwenMoEDraftMixin, Qwen3DSparkModel):
    """DSpark draft with the ``qwen3_5_moe`` FFN: the architecture SpecForge's MoE exports name."""


# The served MoE FFN per moe_preset; the equivalence gate builds one layer through it.
SPARSE_BLOCKS = {"qwen3_5_moe": QwenMoESparseBlock}

EntryClass = [DFlashMoEDraftModel, Qwen3MoeDSparkModel]
