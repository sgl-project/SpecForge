"""How a served target turns its captured hidden state into logits.

SpecForge recomputes target logits from the post-norm final hidden state that
capture returns, using the checkpoint's LM head. SGLang changes that
projection for some architectures, and SpecForge has to reproduce each change
on the side that sees it:

- Head transforms change the LM head that SGLang's draft paths borrow
  (DSpark, DFlash2, Domino): the muP division by
  ``logits_mup_width_multiplier`` (which those paths apply to the draft
  hidden), and load-time head weight rewrites (Grok's scale, Baichuan2 and
  Bailing normalization). They are folded into the frozen head shared by the
  draft and the teacher, so draft logits match serving.
- ``logit_scale`` and ``softcap`` exist only in the target's own logits path
  (``LogitsProcessor``); no draft path reproduces them. They apply to teacher
  logits alone, so soft targets and teacher metrics see the distribution the
  target actually serves.

Served target logits are ``softcap(logit_scale * h @ fold_into_head(W).T)``.
Entries mirror SGLang's model files, keyed like SGLang's dispatch on
``config.architectures[0]``.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetLogitTransform:
    """A target's logit transform, split by who must reproduce it."""

    head_divisor: float = 1.0
    head_weight_scale: float = 1.0
    head_weight_normalize_dim: Optional[int] = None
    head_weight_normalize_eps: float = 1e-12
    logit_scale: float = 1.0
    softcap: Optional[float] = None
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("head_divisor", "head_weight_scale", "logit_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if self.softcap is not None and (
            not math.isfinite(self.softcap) or self.softcap <= 0
        ):
            raise ValueError(f"softcap must be finite and > 0, got {self.softcap}")

    @property
    def head_is_identity(self) -> bool:
        return (
            self.head_divisor == 1.0
            and self.head_weight_scale == 1.0
            and self.head_weight_normalize_dim is None
        )

    @property
    def teacher_is_identity(self) -> bool:
        return self.logit_scale == 1.0 and self.softcap is None

    def fold_into_head(self, weight: torch.Tensor) -> torch.Tensor:
        """The LM head weight SGLang's draft paths effectively project through."""
        if self.head_is_identity:
            return weight
        folded = weight.float()
        if self.head_weight_normalize_dim is not None:
            folded = F.normalize(
                folded,
                p=2,
                dim=self.head_weight_normalize_dim,
                eps=self.head_weight_normalize_eps,
            )
        if self.head_weight_scale != 1.0:
            folded = folded * self.head_weight_scale
        if self.head_divisor != 1.0:
            folded = folded / self.head_divisor
        return folded.to(weight.dtype)

    def apply_teacher(self, logits: torch.Tensor) -> torch.Tensor:
        """Map shared-head logits to the target's served logits.

        Computed in fp32 and returned in the input dtype, so a scaled target's
        teacher costs no more memory than an unscaled one.
        """
        if self.teacher_is_identity:
            return logits
        served = logits.float()
        if self.logit_scale != 1.0:
            served = served * self.logit_scale
        if self.softcap is not None:
            served = torch.tanh(served / self.softcap) * self.softcap
        return served.to(logits.dtype)


IDENTITY = TargetLogitTransform()


def _language_configs(config: Any) -> tuple:
    """The language-model config first, then the top-level config."""
    configs = []
    for name in ("text_config", "llm_config"):
        candidate = getattr(config, name, None)
        if candidate is not None:
            configs.append(candidate)
    configs.append(config)
    return tuple(configs)


def _field(config: Any, name: str, default: Any = None) -> Any:
    for candidate in _language_configs(config):
        value = getattr(candidate, name, None)
        if value is not None:
            return value
    return default


def _softcap(value: Any) -> Optional[float]:
    # LogitsProcessor disables a missing, zero or negative softcap.
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _generic(config: Any, **fields: Any) -> TargetLogitTransform:
    """Fields SGLang honors on every architecture.

    ``LogitsProcessor`` reads ``final_logit_softcapping`` for every model, and
    the DSpark/DFlash workers divide draft hidden by the target's
    ``logits_mup_width_multiplier`` whenever it is declared (inkling.py divides
    the target's own hidden the same way).
    """
    multiplier = _field(config, "logits_mup_width_multiplier")
    fields.setdefault("head_divisor", float(multiplier) if multiplier else 1.0)
    fields.setdefault("softcap", _softcap(_field(config, "final_logit_softcapping")))
    if fields.get("logit_scale") is None:
        fields["logit_scale"] = 1.0
    fields["logit_scale"] = float(fields["logit_scale"])
    sources = [
        f"{name}={value:g}"
        for name, value in fields.items()
        if isinstance(value, float)
        and value != getattr(IDENTITY, name)
        and name != "head_weight_normalize_eps"
    ]
    if fields.get("head_weight_normalize_dim") is not None:
        sources.append(f"normalize(dim={fields['head_weight_normalize_dim']})")
    return TargetLogitTransform(source=", ".join(sources), **fields)


def _minicpm(config: Any) -> TargetLogitTransform:
    # minicpm.py / minicpm3.py: hidden / (hidden_size / dim_model_base).
    dim_model_base = _field(config, "dim_model_base")
    if dim_model_base is None:
        return _generic(config)
    return _generic(
        config,
        logit_scale=float(dim_model_base) / float(_field(config, "hidden_size")),
    )


def _logit_scale(name: str, default: Optional[float] = None):
    # LogitsProcessor(config, logit_scale=config.<name>).
    def resolve(config: Any) -> TargetLogitTransform:
        return _generic(config, logit_scale=_field(config, name, default))

    return resolve


def _granite(config: Any) -> TargetLogitTransform:
    # granite*.py: logit_scale = 1 / logits_scaling.
    scaling = _field(config, "logits_scaling")
    return _generic(config, logit_scale=1.0 / float(scaling) if scaling else None)


def _teleflm(config: Any) -> TargetLogitTransform:
    # teleflm.py: logits * output_mult / mup_scale_factor when use_mup.
    if not _field(config, "use_mup", False):
        return _generic(config)
    return _generic(
        config,
        logit_scale=float(_field(config, "output_mult"))
        / float(_field(config, "mup_scale_factor")),
    )


def _muse_glimmer(config: Any) -> TargetLogitTransform:
    # muse_glimmer.py: LogitsProcessor(config, logit_scale=output_multiplier),
    # softcap from output_soft_cap_temp (flat) or the HF text config's
    # final_logit_softcapping (nested; see configs/muse_glimmer.py).
    if getattr(config, "text_config", None) is not None:
        softcap = _field(config, "final_logit_softcapping")
    else:
        softcap = getattr(config, "output_soft_cap_temp", 20.0)
        if softcap is None:
            softcap = getattr(config, "final_logit_softcapping", None)
    return _generic(
        config,
        logit_scale=_field(config, "output_multiplier", 0.19611613513818404),
        softcap=_softcap(softcap),
    )


def _muse_spark(config: Any) -> TargetLogitTransform:
    # muse_spark.py: MuseSpark1xLogitsProcessor(config, logit_scale=output_multiplier);
    # MuseSparkTextConfig defaults the softcap to 20.
    return _generic(
        config,
        logit_scale=_field(config, "output_multiplier", 0.17677669529663687),
        softcap=_softcap(_field(config, "final_logit_softcapping", 20.0)),
    )


def _beam(config: Any) -> TargetLogitTransform:
    # beam.py: generic softcap; BeamConfig defaults it to 30.
    return _generic(
        config, softcap=_softcap(_field(config, "final_logit_softcapping", 30.0))
    )


def _grok(config: Any) -> TargetLogitTransform:
    # grok.py load_weights: lm_head = fp32(W) * output_multiplier_scale.
    scale = _field(config, "output_multiplier_scale")
    if scale is None:
        raise ValueError("Grok targets require output_multiplier_scale")
    return _generic(config, head_weight_scale=float(scale))


def _baichuan(config: Any) -> TargetLogitTransform:
    # baichuan.py load_weights: Baichuan2 (vocab 125696) normalizes head rows.
    if int(_field(config, "vocab_size", 0)) != 125696:
        return _generic(config)
    return _generic(config, head_weight_normalize_dim=1)


def _bailing(config: Any) -> TargetLogitTransform:
    # bailing_moe.py load_weights: norm_head normalizes each hidden column of
    # an untied head across the vocabulary.
    if not _field(config, "norm_head", False) or _field(
        config, "tie_word_embeddings", False
    ):
        return _generic(config)
    return _generic(config, head_weight_normalize_dim=0, head_weight_normalize_eps=1e-7)


_RESOLVERS: Dict[str, Callable[[Any], TargetLogitTransform]] = {
    "MiniCPMForCausalLM": _minicpm,
    "MiniCPMSALAForCausalLM": _minicpm,
    "MiniCPM3ForCausalLM": _minicpm,
    "FalconH1ForCausalLM": _logit_scale("lm_head_multiplier"),
    "TeleFLMForCausalLM": _teleflm,
    "CohereForCausalLM": _logit_scale("logit_scale"),
    "Cohere2ForCausalLM": _logit_scale("logit_scale"),
    "Cohere2MoeForCausalLM": _logit_scale("logit_scale", 0.0625),
    "Cohere2VisionForConditionalGeneration": _logit_scale("logit_scale", 0.0625),
    "GraniteForCausalLM": _granite,
    "GraniteSWAForCausalLM": _granite,
    "GraniteMoeForCausalLM": _granite,
    "GraniteMoeSharedForCausalLM": _granite,
    "GraniteMoeSWAForCausalLM": _granite,
    "GraniteMoeHybridForCausalLM": _granite,
    "HunYuanMoEV1ForCausalLM": _logit_scale("logit_scale"),
    "HunYuanDenseV1ForCausalLM": _logit_scale("logit_scale"),
    "IQuestQ1ForCausalLM": _logit_scale("logit_scale"),
    "KimiK3ForConditionalGeneration": _logit_scale("logit_scale"),
    "KimiK3LinearForCausalLM": _logit_scale("logit_scale"),
    "KimiLinearForCausalLM": _logit_scale("logit_scale"),
    "SolarForCausalLM": _logit_scale("logit_scale"),
    "MuseGlimmerForCausalLM": _muse_glimmer,
    "MuseGlimmerForConditionalGeneration": _muse_glimmer,
    "MuseSparkForConditionalGeneration": _muse_spark,
    "MuseSparkTextForCausalLM": _muse_spark,
    "BeamForCausalLM": _beam,
    "Grok1ForCausalLM": _grok,
    "Grok1ModelForCausalLM": _grok,
    "BaichuanForCausalLM": _baichuan,
    "BailingMoEForCausalLM": _bailing,
    "BailingMoeForCausalLM": _bailing,
    "BailingMoeV2ForCausalLM": _bailing,
    "SarvamMoEForCausalLM": _bailing,
    "BailingMMNativeForConditionalGeneration": _bailing,
    "BailingMM2NativeForConditionalGeneration": _bailing,
}

# Fields that change served logits only on the architectures above. On any
# other architecture SGLang ignores them; warn in case a new model uses one.
_ARCHITECTURE_FIELDS = (
    "logit_scale",
    "logits_scaling",
    "output_multiplier",
    "output_multiplier_scale",
    "output_soft_cap_temp",
    "lm_head_multiplier",
    "norm_head",
)


def resolve_target_logit_transform(config: Any) -> TargetLogitTransform:
    """Return the logit transform SGLang applies when serving ``config``."""
    architectures = getattr(config, "architectures", None) or ()
    architecture = architectures[0] if architectures else None
    resolver = _RESOLVERS.get(architecture)
    if resolver is not None:
        return resolver(config)
    for name in _ARCHITECTURE_FIELDS:
        value = _field(config, name)
        if value not in (None, False, 0, 0.0, 1, 1.0):
            logger.warning(
                "Target architecture %s sets %s=%r, which SGLang applies only on "
                "specific architectures; SpecForge assumes it is ignored. Add the "
                "architecture to logit_transform._RESOLVERS if SGLang uses it.",
                architecture,
                name,
                value,
            )
    return _generic(config)


__all__ = [
    "IDENTITY",
    "TargetLogitTransform",
    "resolve_target_logit_transform",
]
