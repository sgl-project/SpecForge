from abc import abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn

from .base import TargetEngine
from .capture_policy import DFlashCapturePolicy, DFlashTargetOutput

# NOTE: the capture/load implementations live in
# ``capture_policy.DFlashCapturePolicy``, shared with the generic per-backend
# engines. The classes below keep the existing hierarchy and delegate.

_DFLASH = DFlashCapturePolicy()


class DFlashTargetEngine(TargetEngine):
    """DFlash target engine — the algorithm ABC over a frozen target backend.

    DFlash captures the concatenated hidden states of an arbitrary list of
    target layers (``set_capture_layers``) and trains on hard real-token labels,
    so — unlike EAGLE3 — there is no target distribution / vocab map. The generic
    :meth:`TargetEngine.capture` hook dispatches to ``generate_dflash_data``, so
    the extraction is byte-identical to the pre-Phase-B path.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetEngine":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training."""

    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> DFlashTargetOutput:
        """Generic extraction entry point (see :meth:`TargetEngine.capture`).

        Dispatches to the DFlash-specific ``generate_dflash_data``. DFlash takes
        no extra extraction kwargs, so any are ignored.
        """
        return self.generate_dflash_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        """Set which layers' hidden states to capture (TargetEngine hook)."""
        self.capture_layer_ids = layer_ids


class SGLangDFlashTargetEngine(DFlashTargetEngine):

    backend = "sglang"

    def __init__(self, backend):  # backend: sglang_backend.SGLangCaptureBackend
        super().__init__()  # capture_layer_ids = None
        self._backend = backend

    @property
    def model_runner(self):
        """Kept for back-compat: the underlying sglang ModelRunner."""
        return self._backend.model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDFlashTargetEngine":
        # Lazy import so `import specforge` still works without the pinned sglang.
        from .sglang_backend import SGLangCaptureBackend

        backend = SGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **_DFLASH.spec.sglang_build_kwargs,
            **kwargs,
        )
        return cls(backend)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)  # records self.capture_layer_ids
        # Some target models expose set_eagle3_layers_to_capture; guard on it.
        self._backend.set_eagle3_capture_layers(layer_ids, if_supported=True)

    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        return _DFLASH.sglang_capture(
            self._backend, input_ids, attention_mask, loss_mask
        )


class HFDFlashTargetEngine(DFlashTargetEngine):

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
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetEngine":
        return cls(
            _DFLASH.hf_load(
                pretrained_model_name_or_path,
                torch_dtype,
                device,
                cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        )

    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        return _DFLASH.hf_capture(
            self.model, self.capture_layer_ids, input_ids, attention_mask, loss_mask
        )


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetEngine:
    if backend == "sglang":
        return SGLangDFlashTargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


# --- Back-compat aliases (pre-Phase-B names) -------------------------------
DFlashTargetModel = DFlashTargetEngine
SGLangDFlashTargetModel = SGLangDFlashTargetEngine
HFDFlashTargetModel = HFDFlashTargetEngine
