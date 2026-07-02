import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_device_mesh, get_tp_group
from specforge.utils import padding

from .base import TargetEngine

# NOTE (Phase B2): this module no longer imports sglang internals. The
# SGLang-version-pinned capture path (ServerArgs / ModelConfig / SGLangRunner +
# the extend/capture forward) lives entirely in
# ``sglang_backend.SGLangCaptureBackend``; the SGLang engine below composes it
# (imported lazily inside ``from_pretrained``). A sglang bump touches only that
# module — the engine and ``import specforge`` are sglang-version-agnostic.

logger = logging.getLogger(__name__)


@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_states: Optional[torch.Tensor] = None


class Eagle3TargetEngine(TargetEngine):
    """EAGLE3 target engine — the algorithm ABC over a frozen target backend.

    Offers a layer of abstraction for the target model backend. The user can
    choose different backends to suit their needs:
    1. SGLang backend: for the mainstream model support with the fastest inference speed
    2. HuggingFace backend: for models that are not supported by SGLang but can be loaded by HuggingFace.
    3. Custom backend: for models with customized architecture and inference plan.

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
        """
        Initialize the target model backend from a pretrained model path.
        """

    @abstractmethod
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        """
        Generate the eagle3 data from the target model.
        """

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
        """
        Set the layers to capture the aux hidden states from the target model outputs.
        """
        if aux_hidden_states_layers is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
            else:
                raise ValueError(
                    f"Failed to set aux hidden states layers as model config {self.model.config} does not have num_hidden_layers"
                )
            aux_hidden_states_layers = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers for EAGLE3"


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
        """
        Initialize the HuggingFace target model backend from a pretrained model path.
        """
        tp_size = get_tp_group().size()

        if tp_size > 1:
            device_kwargs = {
                "tp_plan": "auto",
                "tp_size": tp_size,
                "device_mesh": get_tp_device_mesh(),
            }
        else:
            device_kwargs = {
                "device_map": device,
            }

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **device_kwargs,
            **kwargs,
        )
        return cls(target_model)

    def _get_transformer_layers(self):
        """
        Helper to find the module list containing the transformer layers.
        Adapts to common architectures (Llama, Qwen, Mistral, OPT, etc.)
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        else:
            raise ValueError(
                "Could not locate transformer layers in the model architecture to register hooks."
            )

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        """
        Optimized HF backend:
        Instead of returning all hidden states (memory heavy), we use forward hooks
        to capture only the specific layers required by Eagle3.
        """
        if kwargs:
            logger.debug(f"unused kwargs {list(kwargs.keys())}")

        captured_states = {}
        handles = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                # HF outputs for layers are usually tuples (hidden_states, present_key_value, ...)
                # We only need the hidden_states (first element)
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured_states[layer_idx] = hidden

            return hook

        # Locate the transformer layers ModuleList
        layers = self._get_transformer_layers()

        target_indices = self.aux_hidden_states_layers

        # Register hooks
        for idx in target_indices:
            # Ensure index is within bounds
            if 0 <= idx < len(layers):
                handles.append(layers[idx].register_forward_hook(get_hook(idx)))
            else:
                raise ValueError(
                    f"Layer index {idx} out of bounds for model with {len(layers)} layers."
                )

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                output_router_logits=False,
                use_cache=False,
            )
            target = outputs.logits
        finally:
            # Always remove hooks to prevent memory leaks or side effects on subsequent calls
            for handle in handles:
                handle.remove()

        # Verify we captured everything
        if len(captured_states) != 3:
            raise RuntimeError(
                f"Expected to capture 3 layers, but captured {len(captured_states)}"
            )

        # Extract in the correct order
        hidden_states0 = captured_states[target_indices[0]]
        hidden_states1 = captured_states[target_indices[1]]
        hidden_states2 = captured_states[target_indices[2]]

        hidden_states = torch.cat(
            (hidden_states0, hidden_states1, hidden_states2), dim=-1
        )

        # apply pading
        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(target.device)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            wrap_eagle3_logits=True,
            return_full_logits=False,
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

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
        shard_returns: bool = False,
        **kwargs,
    ) -> Eagle3TargetOutput:
        """
        return:
            data_for_draft: List[Dict[str, torch.Tensor]] of draft_batch_size, draft_micro_batch_size = 1
                - input_ids: (1, seq_len)
                - attention_mask: (1, seq_len)
                - loss_mask: (1, seq_len)
                - target: (1, seq_len, vocab_size) or (1, seq_len, hidden_size)
                - hidden_states: (1, seq_len, hidden_size)
                - pixel_values: (patch_len, patch_width)
                - image_grid_thw (batch_size, 3)
        """
        if kwargs:
            logger.debug(f"unused kwargs {list(kwargs.keys())}")

        if is_vlm:
            data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list = (
                self.extend_eagle3_vlm(
                    input_ids,
                    attention_mask,
                    loss_mask,
                    return_last_hidden_states=False,
                    return_logits=True,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            )
        else:
            data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list = (
                self.extend_eagle3(
                    input_ids,
                    attention_mask,
                    loss_mask,
                    return_last_hidden_states=False,
                    return_logits=True,
                    shard_returns=shard_returns,
                )
            )
        aux_hidden_states_out = []
        target_out = []
        loss_mask_out = []
        attention_mask_out = []
        input_ids_out = []
        last_hidden_states_out = []

        for idx, (data, logits, aux_hidden_states, last_hidden_states) in enumerate(
            zip(
                data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list
            )
        ):
            if aux_hidden_states is not None:
                aux_hidden_states_out.append(aux_hidden_states.unsqueeze(0))
                loss_mask_out.append(data[2])
                attention_mask_out.append(data[1])
                input_ids_out.append(data[0])

            # when generating hidden states for offline training, we don't compute logits and only keep the last_hidden_states
            # when training online, we don't keep the last_hidden_states and only keep the logits
            if logits is not None:
                target_out.append(logits.unsqueeze(0))

            if last_hidden_states is not None:
                last_hidden_states_out.append(last_hidden_states.unsqueeze(0))

        aux_hidden_states_out = torch.cat(aux_hidden_states_out, dim=0)

        loss_mask_out = torch.cat(loss_mask_out, dim=0)
        attention_mask_out = torch.cat(attention_mask_out, dim=0)
        input_ids_out = torch.cat(input_ids_out, dim=0)

        if target_out:
            target_out = torch.cat(target_out, dim=0)
        else:
            target_out = None

        if last_hidden_states_out:
            last_hidden_states_out = torch.cat(last_hidden_states_out, dim=0)
        else:
            last_hidden_states_out = None

        if target_out is not None:
            target_out = padding(target_out, left=False)
        input_ids_out = padding(input_ids_out, left=False)
        loss_mask_out = loss_mask_out[..., None]

        return Eagle3TargetOutput(
            hidden_states=aux_hidden_states_out,
            target=target_out,
            loss_mask=loss_mask_out,
            input_ids=input_ids_out,
            attention_mask=attention_mask_out,
            last_hidden_states=last_hidden_states_out,
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

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        if kwargs:
            logger.debug(f"unused kwargs {list(kwargs.keys())}")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            layers_to_output_hidden_states=self.aux_hidden_states_layers,
            use_cache=False,
        )

        # For custom backends, the model implementation is responsible for only
        # returning the requested layers in `outputs.hidden_states`.
        hidden_states = torch.cat(outputs.hidden_states, dim=-1)

        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(target.device)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class SGLangServerEagle3TargetEngine(Eagle3TargetEngine):
    """Live frozen-target SGLang *server* engine (O1.3, W3) — reforward transport.

    The O1.3 capture spike (July 2026, sglang dev, H200) settled the transport
    depth: a stock SGLang server exposes per-request ``return_hidden_states``
    but returns the LAST layer only — the eagle3 aux three-layer capture is not
    reachable over the stock HTTP surface (TorchSpec closed the same gap with a
    17-file engine patch; SpecForge's plan is to upstream a capture API into
    sglang instead — see the spike note in ``docs/roadmap/online-disaggregation.md``).

    Until that API lands, this engine implements the **reforward transport**,
    patch-free on a stock server:

    1. the live server does what only it can do — the DECODING — via
       ``POST /generate`` with raw ``input_ids`` (``--skip-tokenizer-init``
       compatible), greedy by default so the frozen-target stream is
       reproducible;
    2. an in-process capture engine over the SAME weights (hf / sglang /
       custom — the existing, extraction-gate-validated backends) runs ONE
       extend over ``[prompt + completion]`` to produce aux + target features.

    The training-side contract is byte-identical to every other backend:
    ``capture()`` returns a normal :class:`Eagle3TargetOutput`, so
    RolloutWorker → FeatureStore → SampleRef and the trainer see nothing new.
    A future engine-side transport (upstreamed capture API) replaces step 2
    without touching callers.
    """

    backend = "sglang_server"

    def __init__(
        self,
        capture_engine: Eagle3TargetEngine,
        base_url: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        timeout_s: float = 300.0,
    ) -> None:
        self.capture_engine = capture_engine
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        # Mirror the capture contract surface (adapters/launch read it off the
        # engine to build + verify CaptureConfig).
        self.aux_hidden_states_layers = getattr(
            capture_engine, "aux_hidden_states_layers", None
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        *,
        base_url: str = None,
        capture_backend: str = "hf",
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        timeout_s: float = 300.0,
        **kwargs,
    ) -> "SGLangServerEagle3TargetEngine":
        if not base_url:
            raise ValueError(
                "sglang_server engine needs base_url=<http://host:port> of a "
                "running SGLang server over the same weights as "
                f"{pretrained_model_name_or_path!r} (launch with "
                "--skip-tokenizer-init to feed raw input_ids)."
            )
        inner = get_eagle3_target_model(
            pretrained_model_name_or_path,
            backend=capture_backend,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
        return cls(
            inner,
            base_url,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            timeout_s=timeout_s,
        )

    # aux-layer selection delegates to the capture engine (same weights) and is
    # mirrored here so the adapter's recorded-vs-requested verification sees it.
    def set_aux_hidden_states_layers(self, aux_hidden_states_layers=None) -> None:
        self.capture_engine.set_aux_hidden_states_layers(aux_hidden_states_layers)
        self.aux_hidden_states_layers = getattr(
            self.capture_engine, "aux_hidden_states_layers", aux_hidden_states_layers
        )

    def health(self) -> bool:
        import requests

        try:
            return (
                requests.get(f"{self.base_url}/health", timeout=10).status_code == 200
            )
        except requests.RequestException:
            return False

    def _server_generate(self, prompt_rows: List[List[int]], **sampling) -> List[List[int]]:
        import requests

        params = {
            "temperature": sampling.get("temperature", self.temperature),
            "max_new_tokens": sampling.get("max_new_tokens", self.max_new_tokens),
        }
        resp = requests.post(
            f"{self.base_url}/generate",
            json={"input_ids": prompt_rows, "sampling_params": params},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, dict):
            out = [out]
        return [row["output_ids"] for row in out]

    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Eagle3TargetOutput:
        """Live decode on the server, then one capture extend over the result.

        ``input_ids``/``attention_mask`` describe the PROMPTS (right-padded);
        the returned features cover ``[prompt + completion]`` with the loss
        masked to the generated region — the online eagle3 convention (the
        model learns to draft the frozen target's own continuations).
        """
        device = input_ids.device
        prompt_rows = [
            row[mask.bool()].tolist()
            for row, mask in zip(input_ids.cpu(), attention_mask.cpu())
        ]
        # Split kwargs: sampling knobs go to the server; everything else
        # (e.g. shard_returns) belongs to the inner capture call — dropping
        # them would silently break the per-backend capture contract.
        sampling = {
            k: kwargs.pop(k) for k in ("temperature", "max_new_tokens") if k in kwargs
        }
        completions = self._server_generate(prompt_rows, **sampling)

        extended, ext_attn, ext_loss = [], [], []
        for prompt, completion in zip(prompt_rows, completions):
            seq = prompt + list(completion)
            extended.append(torch.tensor(seq, dtype=torch.long))
            ext_attn.append(torch.ones(len(seq), dtype=torch.long))
            row_loss = torch.zeros(len(seq), dtype=torch.long)
            row_loss[len(prompt) :] = 1  # train on the generated region
            # The capture backends left-shift target/input_ids (padding
            # left=False): the row's FINAL position has no next-token teacher,
            # so it must not carry loss — the offline pipeline's convention
            # (preprocessing zeroes loss_mask[-1]) holds here too.
            row_loss[-1] = 0
            ext_loss.append(row_loss)

        pad = torch.nn.utils.rnn.pad_sequence
        ext_ids = pad(extended, batch_first=True, padding_value=0).to(device)
        ext_attn = pad(ext_attn, batch_first=True, padding_value=0).to(device)
        ext_loss = pad(ext_loss, batch_first=True, padding_value=0).to(device)

        return self.capture_engine.capture(
            input_ids=ext_ids, attention_mask=ext_attn, loss_mask=ext_loss, **kwargs
        )


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
# The de-EAGLE3 rename (``*TargetModel`` -> ``*TargetEngine``, ``Eagle3TargetModel``
# -> ``Eagle3TargetEngine``) is import-compatible: existing scripts, tests and
# ``specforge.modeling`` re-exports keep importing the old names. These aliases
# are removed once callers migrate (tracked in the Phase E model-layout move).
Eagle3TargetModel = Eagle3TargetEngine
HFEagle3TargetModel = HFEagle3TargetEngine
SGLangEagle3TargetModel = SGLangEagle3TargetEngine
CustomEagle3TargetModel = CustomEagle3TargetEngine
