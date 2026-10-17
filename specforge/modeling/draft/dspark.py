# coding=utf-8
"""DSpark draft model: DFlash backbone + EAGLE-style Markov and confidence heads.

DSpark shares SpecForge's DFlash block-diffusion drafter (dual-source KV
injection via :class:`DFlashDraftModel`, anchor sampling, MASK-token noise
stream) and adds two heads on top:

  - Markov head: a low-rank learned bigram bias added to the draft logits,
    conditioned on the (teacher-forced) previous token. Improves the per-token
    distribution without touching the backbone.
  - Confidence head (AcceptRatePredictor): predicts a per-draft-position
    acceptance probability, trained against the empirical draft-vs-target
    accept rate (used at inference time for adaptive block length).

Ported from TorchSpec PR #129 (``torchspec/models/draft/dspark.py``). The Markov
/ confidence modeling code is adapted from DeepSeek's DeepSpec
(``deepspec/modeling/dspark/{markov_head,common}.py``, MIT License).

SpecForge differences vs TorchSpec (load-bearing):
  - There is no ``DFlashConfig``; SpecForge's :class:`DFlashDraftModel` uses a
    plain ``Qwen3Config`` plus a ``config.dflash_config`` dict. So
    :class:`DSparkConfig` subclasses ``Qwen3Config`` and declares the DSpark
    fields as top-level attributes; DFlash-carried fields (``block_size``,
    ``num_target_layers``, ``dflash_config``) stay as before.
  - The draft model has no ``embed_tokens`` of its own (the embedding lives on
    the target and is passed into the online wrapper), and the context
    projection is ``self.fc`` (not ``context_proj``). The heads only depend on
    ``config.hidden_size`` / ``config.vocab_size``, so this does not matter for
    construction.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel


class DSparkConfig(Qwen3Config):
    """Configuration for the DSpark draft model.

    Extends ``Qwen3Config`` (SpecForge's DFlash draft is config-light and reads a
    plain ``Qwen3Config``). DSpark-specific fields are declared here; the
    DFlash-carried fields (``block_size``, ``num_target_layers``, and the nested
    ``dflash_config`` dict holding ``target_layer_ids`` / ``mask_token_id``) are
    consumed by the :class:`DFlashDraftModel` base ``__init__`` and must be
    present on the config object before constructing the model.
    """

    model_type = "dspark"

    def __init__(
        self,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov


class VanillaMarkov(nn.Module):
    """Low-rank learned bigram bias added to the draft logits.

    Adapted from DeepSpec's ``deepspec/modeling/dspark/markov_head.py``.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        assert (
            self.markov_rank > 0
        ), f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids)


class AcceptRatePredictor(nn.Module):
    """Per-position acceptance-probability predictor (a single linear head).

    Adapted from DeepSpec's ``deepspec/modeling/dspark/common.py``.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None

    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=config.vocab_size, markov_rank=markov_rank)
    raise NotImplementedError(
        f"markov_head_type={markov_head_type!r} is not supported yet; only 'vanilla' "
        "is implemented as it is recommended by the authors."
    )


class DSparkDraftModel(DFlashDraftModel):
    """DSpark draft network: DFlash backbone + Markov / confidence heads."""

    config_class = DSparkConfig

    def __init__(self, config) -> None:
        super().__init__(config)

        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )

        self.markov_head = build_markov_head(config)

        self.confidence_head: Optional[nn.Module] = None
        if getattr(config, "enable_confidence_head", False):
            conf_input_dim = config.hidden_size
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "confidence_head_with_markov=True requires a Markov head "
                        "(markov_rank > 0)."
                    )
                conf_input_dim += self.markov_rank
            self.confidence_head = AcceptRatePredictor(conf_input_dim)
