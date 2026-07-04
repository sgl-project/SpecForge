import logging
from abc import abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn

from .base import TargetEngine
from .capture_policy import Eagle3CapturePolicy, Eagle3TargetOutput

# NOTE: the capture/load implementations for every backend live in
# ``capture_policy.Eagle3CapturePolicy`` — shared with the generic
# per-backend engines (``hf.py`` / ``sglang.py`` / ``custom.py``). The classes
# below keep the existing hierarchy, tags and method surfaces (scripts,
# adapters and tests import them) and delegate every body to the one policy.

logger = logging.getLogger(__name__)

_EAGLE3 = Eagle3CapturePolicy()


class Eagle3TargetEngine(TargetEngine):
    """EAGLE3 target engine — the algorithm ABC over a frozen target backend.

    EAGLE3 captures three *aux* hidden-state layers plus (optionally) target
    logits. The generic :meth:`capture` / :meth:`set_capture_layers` hooks from
    :class:`TargetEngine` are thin dispatchers onto the EAGLE3-specific
    ``generate_eagle3_data`` / ``set_aux_hidden_states_layers`` below, so the
    extraction is byte-identical to the pre-Phase-B path.
    """

    def __init__(self):
        self.aux_hidden_states_layers = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Eagle3TargetEngine":
        """Initialize the target model backend from a pretrained model path."""

    @abstractmethod
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        """Generate the eagle3 data from the target model."""

    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        """Generic extraction entry point (see :meth:`TargetEngine.capture`).

        Dispatches to the EAGLE3-specific ``generate_eagle3_data``.
        """
        return self.generate_eagle3_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            **kwargs,
        )

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        """Generic alias for EAGLE3 aux-layer selection (TargetEngine hook)."""
        self.set_aux_hidden_states_layers(layer_ids)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """Set the layers to capture the aux hidden states from the target model outputs."""
        config = self.model.config if aux_hidden_states_layers is None else None
        self.aux_hidden_states_layers = _EAGLE3.resolve_capture_layers(
            config, aux_hidden_states_layers
        )


class HFEagle3TargetEngine(Eagle3TargetEngine):

    backend = "hf"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "HFEagle3TargetEngine":
        return cls(
            _EAGLE3.hf_load(
                pretrained_model_name_or_path, torch_dtype, device, cache_dir, **kwargs
            )
        )

    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        return _EAGLE3.hf_capture(
            self.model,
            self.aux_hidden_states_layers,
            input_ids,
            attention_mask,
            loss_mask,
            **kwargs,
        )


class SGLangEagle3TargetEngine(Eagle3TargetEngine):

    backend = "sglang"

    def __init__(self, backend):  # backend: sglang_backend.SGLangCaptureBackend
        # super().__init__() sets aux_hidden_states_layers = None. The sglang
        # backend records capture layers on the model, not on this attribute
        # (unchanged from before: the adapter reads it and gets None for sglang).
        super().__init__()
        self._backend = backend

    @property
    def model_runner(self):
        """Kept for back-compat: the underlying sglang ModelRunner."""
        return self._backend.model_runner

    @property
    def hf_config(self):
        return self._backend.hf_config

    @property
    def is_vlm(self) -> bool:
        return self._backend.is_vlm

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangEagle3TargetEngine":
        # Lazy import so `import specforge` still works without the pinned sglang:
        # the entire sglang-version coupling lives in SGLangCaptureBackend.
        from .sglang_backend import SGLangCaptureBackend

        backend = SGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **_EAGLE3.spec.sglang_build_kwargs,
            **kwargs,
        )
        return cls(backend)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        self._backend.set_eagle3_capture_layers(aux_hidden_states_layers)

    # The extend_eagle3 / extend_eagle3_vlm / get_rope_index forwards live in
    # SGLangCaptureBackend (the version-pinned boundary); the legacy extend /
    # extend_vlm aliases keep the engine's method surface stable while the engine
    # itself imports no sglang internals.
    def extend_eagle3(self, *args, **kwargs):
        return self._backend.extend_eagle3(*args, **kwargs)

    def extend_eagle3_vlm(self, *args, **kwargs):
        return self._backend.extend_eagle3_vlm(*args, **kwargs)

    def extend(self, *args, **kwargs):
        return self.extend_eagle3(*args, **kwargs)

    def extend_vlm(self, *args, **kwargs):
        return self.extend_eagle3_vlm(*args, **kwargs)

    def get_rope_index(self, *args, **kwargs):
        return self._backend.get_rope_index(*args, **kwargs)

    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        return _EAGLE3.sglang_capture(
            self._backend, input_ids, attention_mask, loss_mask, **kwargs
        )


class CustomEagle3TargetEngine(Eagle3TargetEngine):

    backend = "custom"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "CustomEagle3TargetEngine":
        from specforge.modeling.auto import AutoDistributedTargetModel

        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            device=device,
            **kwargs,
        )
        return cls(target_model)

    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        return _EAGLE3.custom_capture(
            self.model,
            self.aux_hidden_states_layers,
            input_ids,
            attention_mask,
            loss_mask,
            **kwargs,
        )


class SGLangServerEagle3TargetEngine(Eagle3TargetEngine):
    """Live frozen-target SGLang *server* engine (cross-node) — W3 / W3′.

    The fourth backend: instead of an in-process ``ModelRunner``, a *frozen*
    target runs as a live SGLang server and streams hidden states (capture into a
    FeatureStore for W3, or inline over HTTP for the light W3′). It is
    ``backend="sglang_server"`` — selectable through the factory — but the live
    capture implementation is gated by the O1.3 throughput spike (see
    ``docs/roadmap/online-disaggregation.md`` §O1.3), so construction raises with an
    actionable message until that lands. The de-EAGLE3 extraction and the domain
    Trainer carry no engine risk and do not depend on this backend.
    """

    backend = "sglang_server"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "sglang_server target engine (live cross-node frozen-target capture) is "
            "not implemented yet; its depth is gated by the O1.3 capture spike "
            "(docs/roadmap/online-disaggregation.md §O1.3). Use backend='sglang' "
            "(in-process) or 'hf' for now."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "SGLangServerEagle3TargetEngine":
        return cls()

    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:  # pragma: no cover - unreachable (ctor raises)
        raise NotImplementedError


def get_eagle3_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Eagle3TargetEngine:
    if backend == "sglang":
        return SGLangEagle3TargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFEagle3TargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "custom":
        return CustomEagle3TargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "sglang_server":
        return SGLangServerEagle3TargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


# --- Back-compat aliases (pre-Phase-B names) -------------------------------
Eagle3TargetModel = Eagle3TargetEngine
HFEagle3TargetModel = HFEagle3TargetEngine
SGLangEagle3TargetModel = SGLangEagle3TargetEngine
CustomEagle3TargetModel = CustomEagle3TargetEngine
