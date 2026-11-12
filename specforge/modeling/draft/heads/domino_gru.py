"""GRU correction over aligned token prefixes and decoder suffix states."""

import math

import torch
from torch import Tensor, nn
from torch.func import functional_call
from transformers import initialization as init

from specforge.config.draft._validation import positive_int
from specforge.config.draft.architecture.domino import GRUHeadSpec


class _PrefixGRU(nn.GRU):
    def reset_draft_parameters(self):
        # Match nn.GRU's uniform initialization while respecting HF load flags.
        bound = 1 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            init.uniform_(parameter, -bound, bound)


def _run_gru(gru: nn.GRU, inputs: Tensor, *, device_type: str) -> Tensor:
    if device_type == "npu" and inputs.dtype == torch.bfloat16:
        # Ascend DynamicGRU does not accept BF16. Keep registered parameters in
        # their original dtype; functional FP16 views retain their gradients.
        parameters = {
            name: parameter.to(dtype=torch.float16)
            for name, parameter in gru.named_parameters()
        }
        output, _ = functional_call(
            gru, parameters, (inputs.to(dtype=torch.float16),), strict=True
        )
        return output.to(dtype=inputs.dtype)
    return gru(inputs)[0]


def _correction_logits(
    gru: nn.GRU,
    projection: nn.Module,
    hidden_states: Tensor,
    prefix_embeddings: Tensor,
    *,
    device_type: str,
) -> Tensor:
    if (
        hidden_states.ndim < 3
        or prefix_embeddings.ndim != hidden_states.ndim
        or prefix_embeddings.shape[:-2] != hidden_states.shape[:-2]
        or hidden_states.shape[-1] != gru.input_size
        or prefix_embeddings.shape[-1] != gru.input_size
        or prefix_embeddings.shape[-2] < 1
        or hidden_states.shape[-2] > prefix_embeddings.shape[-2]
    ):
        raise ValueError(
            "aligned prefix and suffix must share batch/hidden dimensions; prefix must cover the suffix"
        )
    length = prefix_embeddings.shape[-2]
    inputs = prefix_embeddings.reshape(-1, length, gru.input_size)
    states = _run_gru(gru, inputs, device_type=device_type).reshape(
        *hidden_states.shape[:-2], length, gru.hidden_size
    )
    # Earlier prefix tokens contribute recurrent state, but only the aligned
    # suffix receives corrections. This also preserves an empty suffix.
    states = states[..., length - hidden_states.shape[-2] :, :]
    return projection(torch.cat([hidden_states, states], dim=-1))


class DominoGRUHead(nn.Module):
    """The model aligns token prefixes and selects the decoder suffix.

    This head owns no block size, label shift, sampling policy, or base logits.
    It returns only the additive correction for the supplied suffix states.
    """

    def __init__(
        self,
        config: GRUHeadSpec,
        *,
        hidden_size: int,
        vocab_size: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) is not GRUHeadSpec:
            raise TypeError("DominoGRUHead requires GRUHeadSpec")
        positive_int("hidden_size", hidden_size)
        positive_int("vocab_size", vocab_size)
        self.config = config
        options = dict(device=device, dtype=dtype)
        self.prefix_gru = _PrefixGRU(
            input_size=hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bias=False,
            **options,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(
                config.input_width(hidden_size),
                config.projection_size,
                bias=False,
                **options,
            ),
            nn.SiLU(),
            nn.Linear(config.projection_size, vocab_size, bias=False, **options),
        )

    def forward(self, hidden_states: Tensor, prefix_embeddings: Tensor) -> Tensor:
        return _correction_logits(
            self.prefix_gru,
            self.embed_proj,
            hidden_states,
            prefix_embeddings,
            device_type=prefix_embeddings.device.type,
        )
