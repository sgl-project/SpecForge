# coding=utf-8
"""SGLang draft classes for DFlash-family drafters with a Qwen-style MoE FFN.

``Qwen3MoeDSparkModel`` (and ``DFlashMoEDraftModel`` / ``DFlash2MoEDraftModel``)
are SGLang's ``Qwen3DSparkModel`` / ``DFlashDraftModel`` / ``DFlash2DraftModel``
with every decoder layer's dense MLP replaced by SGLang's own MoE block, the
modules SGLang's Qwen2/Qwen3-MoE targets run (``Qwen2MoeSparseMoeBlock``): a
``TopK`` module, a ``FusedMoE`` holding the routed experts and a separate
sigmoid-gated shared expert. Checkpoint tensors load through those modules'
own weight loaders (``FusedMoE.make_expert_params_mapping`` for the per-expert
tensors), so expert-width padding, runner-specific weight layouts, quantised
checkpoints and tensor-parallel sharding are SGLang's business, not ours.

The MoE recipe comes from the export's ``config.json`` as SpecForge writes it
(Qwen keys: ``num_experts``, ``num_experts_per_tok``, ``moe_intermediate_size``,
``shared_expert_intermediate_size``, ``norm_topk_prob``, ``moe_router_bias``,
``shared_expert_gate``). Supported recipe: softmax scoring, renormalised top-k,
no selection bias, no expert groups, no SwiGLU clamp, at most one shared
expert; that is SpecForge's ``qwen3_5_moe`` preset (the Qwen3.8-27B DSpark MoE
drafters). Both the Qwen checkpoint naming (``experts.{i}.gate_proj`` ...)
and the DeepSeek one (``experts.{i}.w1`` ...) are accepted.

Router logits are computed in fp32 (``x W^T + gate.bias``) exactly as the
trainer does, so the experts a token selects match training; only the top-k
selection, renormalisation, expert GEMMs and combine run in SGLang's kernels.
The fused MoE kernel reads its tile configuration from
``E=<experts>,N=<width>,device_name=<GPU>.json`` (see ``patches/sglang/moe_configs``).

Registered through ``SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models``.
Written against SGLang v0.5.19+.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
from sglang.srt.models.dflash import (
    DFlash2DraftModel,
    DFlashDecoderLayer,
    DFlashDraftModel,
)
from sglang.srt.models.dspark import Qwen3DSparkModel
from torch import nn

from .moe_ffn import PRESET_DEFAULTS, routed_expert_count, to_qwen_names

logger = logging.getLogger(__name__)

_EXPERT_TENSOR = re.compile(
    r"^experts\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_LAYER_FFN = re.compile(r"^(?:model\.)?layers\.(?P<idx>\d+)\.mlp\.(?P<key>.+)$")


def _cfg(config, key: str, default):
    value = getattr(config, key, None)
    return default if value is None else value


class QwenSharedExpertMLP(nn.Module):
    """The shared expert as SGLang builds it for Qwen MoE targets
    (``Qwen2MoeMLP``): ``gate_up_proj`` (a ``MergedColumnParallelLinear`` of the
    gate and up projections), fused ``SiluAndMul``, ``down_proj``."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, quant_config, prefix: str
    ):
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
    """``y = FusedMoE(x, TopK(gate(x))) + sigmoid(shared_expert_gate(x)) * shared_expert(x)``
    for one draft layer, built from SGLang's MoE modules (TP1, replicated draft)."""

    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        preset = PRESET_DEFAULTS.get(str(_cfg(config, "moe_preset", "")), {})

        def get(key, default):
            return _cfg(config, key, preset.get(key, default))

        self.hidden_size = int(config.hidden_size)
        self.num_experts = routed_expert_count(config)
        self.top_k = int(_cfg(config, "num_experts_per_tok", 0))
        self.intermediate_size = int(_cfg(config, "moe_intermediate_size", 0))
        if self.num_experts <= 0 or self.top_k <= 0 or self.intermediate_size <= 0:
            raise ValueError(
                "MoE draft config needs num_experts (or n_routed_experts), "
                "num_experts_per_tok and moe_intermediate_size; got "
                f"{self.num_experts}, {self.top_k}, {self.intermediate_size}."
            )
        unsupported = []
        if str(get("scoring_func", "softmax")) != "softmax":
            unsupported.append(f"scoring_func={get('scoring_func', None)}")
        if str(get("topk_method", "greedy")) == "noaux_tc":
            unsupported.append("topk_method=noaux_tc (selection bias)")
        if int(get("n_group", 1)) != 1 or int(get("topk_group", 1)) != 1:
            unsupported.append("expert groups")
        if float(get("routed_scaling_factor", 1.0)) != 1.0:
            unsupported.append(
                f"routed_scaling_factor={get('routed_scaling_factor', None)}"
            )
        if float(get("swiglu_limit", 0.0)) != 0.0:
            unsupported.append(f"swiglu_limit={get('swiglu_limit', None)}")
        if str(_cfg(config, "hidden_act", "silu")) != "silu":
            unsupported.append(f"hidden_act={_cfg(config, 'hidden_act', None)}")
        if unsupported:
            raise ValueError(
                f"{type(self).__name__} serves the Qwen MoE recipe only (softmax, "
                "renormalised top-k, no selection bias, no groups, no SwiGLU clamp); "
                f"this export needs: {', '.join(unsupported)}."
            )
        self.norm_topk_prob = bool(get("norm_topk_prob", True))
        self.router_bias = bool(get("moe_router_bias", False))
        n_shared = int(get("n_shared_experts", 0) or 0)
        if n_shared not in (0, 1):
            raise ValueError("n_shared_experts must be 0 or 1 for MoE drafts")
        self.shared_expert_gate_kind = str(get("shared_expert_gate", "none"))
        if self.shared_expert_gate_kind not in ("none", "sigmoid"):
            raise ValueError(
                f"unknown shared_expert_gate {self.shared_expert_gate_kind!r}; known: none, sigmoid"
            )

        # Router in fp32 like the trainer; ReplicatedLinear only for its loader.
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
            # Read by runner backends that route inside the kernel (flashinfer
            # TRT-LLM): softmax -> top-k -> renormalise.
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            inplace=False,
        )
        self.shared_expert: Optional[QwenSharedExpertMLP] = None
        self.shared_expert_gate: Optional[nn.Linear] = None
        if n_shared:
            self.shared_expert = QwenSharedExpertMLP(
                self.hidden_size,
                int(
                    _cfg(
                        config,
                        "shared_expert_intermediate_size",
                        self.intermediate_size,
                    )
                ),
                quant_config,
                prefix=f"{prefix}.shared_expert",
            )
            if self.shared_expert_gate_kind == "sigmoid":
                self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        self._loaded_experts: Set[Tuple[int, str]] = set()
        self._loaded_params: Set[str] = set()

    def describe(self) -> str:
        shared = "none"
        if self.shared_expert is not None:
            shared = f"{self.shared_expert.down_proj.input_size}" + (
                " (sigmoid-gated)" if self.shared_expert_gate is not None else ""
            )
        return (
            f"{self.num_experts} routed experts, top-{self.top_k}, width "
            f"{self.intermediate_size}, shared={shared}, router_bias={self.router_bias}, "
            f"renorm={self.norm_topk_prob}, runner={type(self.experts.quant_method).__name__}"
        )

    def router_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.gate(x.float())
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        shared = None
        if self.shared_expert is not None:
            shared = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                shared = (
                    torch.sigmoid(self.shared_expert_gate(x).float()).to(shared.dtype)
                    * shared
                )
        y = self.experts(x, self.topk(x, self.router_logits(x)))
        if isinstance(y, tuple):  # some SGLang builds return (out, bias)
            y = y[0]
        if shared is not None:
            y = y + shared
        return y.view(shape)

    # -- loading ---------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load this block's checkpoint entries (names relative to the block, Qwen
        or DeepSeek naming) through the modules' own weight loaders."""
        params = dict(self.named_parameters())
        mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.num_experts,
        )
        for raw_name, tensor in weights:
            name = to_qwen_names(raw_name)
            m = _EXPERT_TENSOR.match(name)
            if m is not None:
                for param_name, weight_name, expert_id, shard_id in mapping:
                    if weight_name not in name:
                        continue
                    # "experts.{i}.gate_proj.weight" -> "experts.w13_weight", as
                    # SGLang's own MoE models do.
                    pname = name.replace(weight_name, param_name)
                    param = params[pname]
                    param.weight_loader(
                        param, tensor, pname, shard_id=shard_id, expert_id=expert_id
                    )
                    self._loaded_experts.add((expert_id, shard_id))
                    break
                else:
                    raise ValueError(
                        f"expert tensor {raw_name!r} is outside num_experts={self.num_experts}"
                    )
                continue
            if name.startswith("shared_expert.gate_proj.") or name.startswith(
                "shared_expert.up_proj."
            ):
                param = params["shared_expert.gate_up_proj.weight"]
                param.weight_loader(param, tensor, 0 if "gate_proj" in name else 1)
                self._loaded_params.add(f"{name}")
                continue
            if name not in params:
                raise ValueError(
                    f"checkpoint FFN entry {raw_name!r} does not map to any parameter of "
                    f"{type(self).__name__} ({sorted(params)[:6]} ...); check that config.json's "
                    "`architectures` matches the export."
                )
            param = params[name]
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, tensor)
            else:
                with torch.no_grad():
                    param.copy_(tensor.to(param.dtype))
            self._loaded_params.add(name)

    def check_loaded(self, where: str = "") -> None:
        """Refuse to serve a partially loaded FFN (an MoE export loaded into a
        class whose config disagrees, a truncated export, ...)."""
        missing: List[str] = []
        for expert_id in range(self.num_experts):
            for shard in ("w1", "w2", "w3"):
                if (expert_id, shard) not in self._loaded_experts:
                    missing.append(f"experts.{expert_id}.{shard}")
        expected = {"gate.weight"}
        if self.router_bias:
            expected.add("gate.bias")
        if self.shared_expert is not None:
            expected |= {
                "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight",
                "shared_expert.down_proj.weight",
            }
            if self.shared_expert_gate is not None:
                expected.add("shared_expert_gate.weight")
        missing += sorted(expected - self._loaded_params)
        if missing:
            shown = ", ".join(missing[:8])
            raise ValueError(
                f"{where or type(self).__name__}: {len(missing)} FFN tensor(s) were not in "
                f"the checkpoint and would serve uninitialised: {shown}"
                f"{' ...' if len(missing) > 8 else ''}"
            )


class QwenMoEDecoderLayer(DFlashDecoderLayer):
    """``DFlashDecoderLayer`` with the dense MLP replaced by :class:`QwenMoESparseBlock`."""

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        layer_id = int(kwargs.get("layer_id", args[0] if args else 0))
        quant_config = kwargs.get("quant_config", args[3] if len(args) > 3 else None)
        prefix = kwargs.get("prefix", args[4] if len(args) > 4 else "")
        # The base layer built a dense MLP the export does not carry; swap it
        # for the MoE block before any weights are loaded.
        del self.mlp
        self.mlp = QwenMoESparseBlock(
            config, layer_id, quant_config, prefix=f"{prefix}.mlp" if prefix else "mlp"
        )


class _QwenMoEDraftMixin:
    """Shared constructor check, logging and strict FFN loading."""

    decoder_layer_cls = QwenMoEDecoderLayer

    def __init__(self, config, *args, **kwargs) -> None:
        if routed_expert_count(config) <= 0:
            raise ValueError(
                f"{type(self).__name__} requires num_experts > 0 in the draft config; "
                "use the dense draft class for dense drafts."
            )
        super().__init__(config, *args, **kwargs)
        logger.info(
            "MoE draft (%s): %s", type(self).__name__, self.layers[0].mlp.describe()
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Every layers.{i}.mlp.* tensor goes to that layer's block; the base
        # loader takes the rest (attention, norms, projector, heads).
        rest: List[Tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            m = _LAYER_FFN.match(name)
            if m is None:
                rest.append((name, tensor))
                continue
            idx = int(m["idx"])
            if idx >= len(self.layers):
                raise ValueError(
                    f"checkpoint has FFN weights for layer {idx} but the draft has {len(self.layers)} layers"
                )
            self.layers[idx].mlp.load_weights([(m["key"], tensor)])
        super().load_weights(iter(rest))
        for idx, layer in enumerate(self.layers):
            layer.mlp.check_loaded(f"{type(self).__name__} layers.{idx}.mlp")


class DFlashMoEDraftModel(_QwenMoEDraftMixin, DFlashDraftModel):
    """DFlash draft with a Qwen-style MoE FFN."""


class DFlash2MoEDraftModel(_QwenMoEDraftMixin, DFlash2DraftModel):
    """DFlash2 draft (grouped convolutions + candidate selector) with a Qwen-style MoE FFN."""


class Qwen3MoeDSparkModel(_QwenMoEDraftMixin, Qwen3DSparkModel):
    """DSpark draft with Qwen3-style attention and a Qwen-style MoE FFN: the
    architecture name SpecForge's Qwen3.8-27B DSpark MoE exports carry."""


class Qwen3MoEDSparkModel(Qwen3MoeDSparkModel):
    """Alias with the capitalisation older normalizers wrote."""


EntryClass = [
    DFlashMoEDraftModel,
    DFlash2MoEDraftModel,
    Qwen3MoeDSparkModel,
    Qwen3MoEDSparkModel,
]
