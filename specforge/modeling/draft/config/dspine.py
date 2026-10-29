"""DSpine injection and refinement settings, validated before model allocation."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, fields

from .base import method_config
from .dflash import DFlashConfig


@dataclass(frozen=True)
class DSpineConfig:
    transfer_rank: int = 1024
    message_size: int = 512
    top_k: int = 16
    ce_weight: float = 0.1
    alignment_weight: float = 0.5
    alignment_warmup_steps: int = 146
    replacement_layers: int = 3
    replacement_probability: float = 0.5
    replacement_hold_ratio: float = 1 / 6
    replacement_end_ratio: float = 1 / 3
    candidate_mass_threshold: float = 1e-4

    def validate(self, hidden_size: int, vocab_size: int) -> None:
        for name in (
            "transfer_rank",
            "message_size",
            "top_k",
            "alignment_warmup_steps",
            "replacement_layers",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"DSpine {name} must be an integer")
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"DSpine {name} must be finite")
        if not 0 < self.transfer_rank <= min(hidden_size, vocab_size - 1):
            raise ValueError(
                "DSpine transfer_rank must be in [1, min(hidden_size, vocab_size - 1)]"
            )
        if self.message_size < 1 or not 1 <= self.top_k <= vocab_size:
            raise ValueError(
                "DSpine requires message_size > 0 and top_k in [1, vocab_size]"
            )
        if not 0 <= self.ce_weight <= 1 or self.alignment_weight < 0:
            raise ValueError(
                "DSpine requires ce_weight in [0, 1] and alignment_weight >= 0"
            )
        if self.alignment_warmup_steps < 0 or self.replacement_layers < 0:
            raise ValueError(
                "DSpine warmup steps and replacement layers must be nonnegative"
            )
        if not 0 <= self.replacement_probability <= 1:
            raise ValueError("DSpine replacement_probability must be in [0, 1]")
        if not 0 <= self.replacement_hold_ratio < self.replacement_end_ratio <= 1:
            raise ValueError(
                "DSpine requires 0 <= replacement_hold_ratio < replacement_end_ratio <= 1"
            )
        if not 0 <= self.candidate_mass_threshold < 1:
            raise ValueError("DSpine candidate_mass_threshold must be in [0, 1)")

    def schedule(self, step: int, total_steps: int | None) -> tuple[float, float]:
        warmup = self.alignment_warmup_steps
        beta = self.alignment_weight * (min(max(step, 0) / warmup, 1) if warmup else 1)
        if not total_steps or total_steps <= 0:
            return beta, 0.0
        progress = max(step, 0) / total_steps
        fraction = (self.replacement_end_ratio - progress) / (
            self.replacement_end_ratio - self.replacement_hold_ratio
        )
        return beta, self.replacement_probability * min(max(fraction, 0), 1)


@dataclass(frozen=True, kw_only=True)
class DSpineDraftConfig(DFlashConfig):
    algorithm = "dspine"
    non_spec_fields = DFlashConfig.non_spec_fields | {"dspine"}
    spec_aliases = {
        field.name: ("dspine_config", field.name) for field in fields(DSpineConfig)
    }

    dspine: DSpineConfig

    @classmethod
    def from_model_config(cls, config):
        config = copy.deepcopy(config)
        if getattr(config, "is_causal", None) is False:
            raise ValueError(
                "DSpine requires causal attention; is_causal=false is unsupported"
            )
        config.is_causal = True
        return super().from_model_config(config)

    @classmethod
    def _variant_fields(cls, config, method, mode, block_size):
        settings = DSpineConfig(**method_config(config, "dspine_config"))
        settings.validate(config.hidden_size, config.vocab_size)
        return {
            **super()._variant_fields(config, method, mode, block_size),
            "dspine": settings,
        }

    def to_model_config(self, config):
        result = super().to_model_config(config)
        result.dflash_config.pop("dspine", None)
        result.dspine_config = asdict(self.dspine)
        return result
