"""EAGLE-3.1 with a learned same-position query and halt gate (arXiv:2611.05724)."""

from collections.abc import Sequence

from torch import Tensor, nn

from ..hf_config import SpecForgeDraftConfig
from ..layers.attention.kv import AttentionKV
from ..layers.inputs import LayerInputs
from .eagle3 import Eagle3DraftModel, Eagle3Output
from .registry import register_draft_model


@register_draft_model
class ParallelReconsiderationDraftModel(Eagle3DraftModel):
    """Reuse the decoder and token head for the one-pass and reconsidered states."""

    def __init__(
        self, config: SpecForgeDraftConfig, *, backend="eager", device=None, dtype=None
    ):
        super().__init__(config, backend=backend, device=device, dtype=dtype)
        width = config.draft.hidden_size
        self.self_query = nn.Linear(width, width, device=device, dtype=dtype)
        self.halt_gate = nn.Linear(width, 1, device=device, dtype=dtype)
        self.post_init()

    def reconsider(
        self,
        hidden: Tensor,
        layer_inputs: Sequence[LayerInputs],
        *,
        kv_prefix: AttentionKV | None = None,
        output_logits: bool = True,
    ) -> Eagle3Output:
        """Refine [batch, sequence, hidden] at the caller's unchanged positions.

        The attention layout must expose the strict committed prefix and the
        newly projected self key at each position. Returned KV is the replacement
        for this position on the reconsidered branch.
        """
        return self._decode_embeddings(
            self.self_query(hidden).to(hidden.dtype),
            hidden,
            layer_inputs,
            kv_prefix=kv_prefix,
            output_logits=output_logits,
        )
