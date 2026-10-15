"""Transformers-based offline capture for targets no serving engine runs locally.

Qwen3.8-Flash-Next has no SGLang path on Ascend and vLLM-Ascend runs it only
on A3, so its features are captured with the Transformers reference model on
one host: a single process places consecutive decoder layers on the visible
accelerators and runs the sequence through them in turn. It is a pipeline
with one stage busy at a time -- slow, but it needs nothing beyond PyTorch.

The backend returns the same ``(data, aux_rows, last_rows)`` triple as the
SGLang backend, so ``scripts/prepare_hidden_states.py`` writes identical files.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from specforge.offline_capture import qwen4_exp

# Modules kept in host memory instead of on an accelerator. Qwen3.8-Flash-Next's
# n-gram table is ~95 GiB in BF16 -- more than one card -- and is only gathered
# from, so its lookup runs on the CPU (the model moves the result back).
_HOST_RESIDENT = ("ple_embedding.ngram_embedding",)


def _accelerator() -> str:
    from specforge.utils import get_device_type

    device_type = get_device_type()
    if device_type == "npu" and not hasattr(torch, "npu"):
        import torch_npu  # noqa: F401
    return device_type


def _module_bytes(module: torch.nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in list(module.parameters()) + list(module.buffers())
    )


def plan_layer_devices(
    layer_bytes: Sequence[int], num_devices: int, head_bytes: int, tail_bytes: int
) -> List[int]:
    """Assign consecutive layers to devices so each holds a near-equal share.

    ``head_bytes`` (embeddings, vision tower) sits on the first device and
    ``tail_bytes`` (final mixer, LM head) on the last; both count toward the
    balance. Returns one device index per layer.
    """

    if num_devices < 1:
        raise ValueError("need at least one device")
    share = (head_bytes + sum(layer_bytes) + tail_bytes) / num_devices
    plan = []
    filled = head_bytes
    for size in layer_bytes:
        # A layer goes where its midpoint falls in the running byte count.
        device = min(num_devices - 1, int((filled + size / 2) // share))
        plan.append(max(device, plan[-1] if plan else 0))
        filled += size
    return plan


def dispatch_layers(model, devices: Sequence[torch.device]) -> Dict[str, object]:
    """Place the model's layers on ``devices`` and route activations between them.

    Each decoder layer gets a pre-hook that moves its tensor inputs to the
    layer's device; the final mixer and head get the same. Host-resident
    modules stay on the CPU, where the model code already sends their inputs.
    """

    inner = qwen4_exp.text_model(model)
    layers = list(inner.layers)
    host_modules = [
        module
        for name, module in model.named_modules()
        if name.endswith(_HOST_RESIDENT)
    ]
    host_ids = {id(module) for module in host_modules}

    def placed_bytes(module):
        return _module_bytes(module) - sum(
            _module_bytes(sub) for sub in module.modules() if id(sub) in host_ids
        )

    head = [inner.embed_tokens]
    visual = getattr(getattr(model, "model", model), "visual", None)
    if visual is not None:
        head.append(visual)
    tail = [
        module
        for module in (
            getattr(inner, "hyper_connection_mixer", None),
            getattr(model, "lm_head", None),
        )
        if module is not None
    ]
    plan = plan_layer_devices(
        [placed_bytes(layer) for layer in layers],
        len(devices),
        sum(placed_bytes(module) for module in head),
        sum(placed_bytes(module) for module in tail),
    )

    def move_inputs(device):
        def hook(_module, args, kwargs):
            def to(value):
                if isinstance(value, torch.Tensor):
                    return value.to(device, non_blocking=True)
                if isinstance(value, tuple):
                    return tuple(to(item) for item in value)
                return value

            return to(args), {key: to(value) for key, value in kwargs.items()}

        return hook

    def place(module, device):
        # Module.to() would recurse into a host-resident table and copy all of
        # it onto one card before anything could move it back, so move each
        # module's own tensors and stop at the host-resident subtrees.
        if id(module) in host_ids:
            return
        module._apply(lambda tensor: tensor.to(device), recurse=False)
        for child in module.children():
            place(child, device)

    for module in head:
        place(module, devices[0])
    rotary = getattr(inner, "rotary_emb", None)
    if rotary is not None:
        place(rotary, devices[0])
    for layer, index in zip(layers, plan):
        place(layer, devices[index])
        layer.register_forward_pre_hook(move_inputs(devices[index]), with_kwargs=True)
    for module in tail:
        place(module, devices[-1])
        module.register_forward_pre_hook(move_inputs(devices[-1]), with_kwargs=True)
    for module in host_modules:
        stray = {
            tensor.device
            for tensor in list(module.parameters()) + list(module.buffers())
            if tensor.device.type != "cpu"
        }
        if stray:
            raise RuntimeError(
                f"host-resident {type(module).__name__} was loaded onto {stray}; "
                "load the target into host memory before dispatching it"
            )

    per_device = [0] * len(devices)
    per_device[0] += sum(placed_bytes(module) for module in head)
    per_device[-1] += sum(placed_bytes(module) for module in tail)
    for layer, index in zip(layers, plan):
        per_device[index] += placed_bytes(layer)
    return {
        "layer_devices": plan,
        "device_gib": [round(size / 2**30, 2) for size in per_device],
        "host_gib": round(sum(_module_bytes(m) for m in host_modules) / 2**30, 2),
    }


class OfflineHFCaptureBackend:
    """Frozen Transformers target used only to materialize offline features."""

    # Rows are captured without their right padding; the preparation script
    # saves each record at its attention-mask length.
    unpadded_rows = True

    def __init__(self, model, *, output_device: torch.device) -> None:
        self.model = model
        self.output_device = output_device
        self.layer_ids: Optional[List[int]] = None
        self.config = model.config

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        num_devices: Optional[int] = None,
        attn_implementation: str = "sdpa",
        fast_paths: bool = True,
        **_unused,
    ) -> "OfflineHFCaptureBackend":
        import transformers
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        if not qwen4_exp.is_qwen4_exp(config):
            raise ValueError(
                "the HF capture backend supports only qwen4_exp targets so far, "
                f"got model_type={config.model_type!r}"
            )
        architecture = (config.architectures or ["Qwen4ExpForConditionalGeneration"])[0]
        model_class = getattr(transformers, architecture)
        # Loads into host memory; dispatch_layers then moves everything but the
        # host-resident tables. Accelerate's own device_map cannot express that:
        # it treats "cpu" as offload and would stream the n-gram table onto the
        # execution device on every forward.
        model = model_class.from_pretrained(
            pretrained_model_name_or_path,
            dtype=torch_dtype if torch_dtype is not None else "auto",
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        ).eval()
        model.requires_grad_(False)

        device_type = _accelerator()
        if device_type == "cpu":
            devices = [torch.device("cpu")]
        else:
            module = getattr(torch, device_type)
            count = num_devices or module.device_count()
            devices = [torch.device(device_type, index) for index in range(count)]
        placement = dispatch_layers(model, devices)
        # print, not logging: the preparation script configures no handler, and
        # the per-device GiB is what an operator checks first on a new host.
        print(f"HF capture placement: {placement}", flush=True)
        if fast_paths:
            installed = qwen4_exp.install_fast_paths(model)
            print(f"HF capture fast paths installed: {installed}", flush=True)
        return cls(model, output_device=devices[0])

    def set_capture_layers(
        self,
        layer_ids: Optional[List[int]] = None,
        *,
        capture_method: str,
    ) -> None:
        if capture_method not in {"dflash", "dspark"}:
            raise ValueError(
                "the HF capture backend writes DFlash-family features only, "
                f"got capture method {capture_method!r}"
            )
        if not layer_ids:
            raise ValueError("the HF capture backend needs explicit capture layers")
        self.layer_ids = list(layer_ids)

    @torch.no_grad()
    def _capture_row(self, input_ids: torch.Tensor):
        sink: Dict[int, torch.Tensor] = {}
        handles = qwen4_exp.hyper_connection_taps(self.model, self.layer_ids, sink)
        try:
            output = qwen4_exp.text_model(self.model)(
                input_ids=input_ids, use_cache=False
            )
        finally:
            for handle in handles:
                handle.remove()
        aux = torch.cat(
            [sink[layer_id].to(self.output_device) for layer_id in self.layer_ids],
            dim=-1,
        )
        return aux[0], output.last_hidden_state[0].to(self.output_device)

    @torch.no_grad()
    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Capture each row on its own, without its right padding.

        Rows run one at a time and only over their real tokens: padding would
        cost target compute (and, given an attention mask, take the indexer
        off its vectorized path), and a pipelined forward gains nothing from
        batching. Each row's states are zero-padded back to the batch length
        so the rows stack; the preparation script trims every saved record to
        its real length (``unpadded_rows``).
        """

        if self.layer_ids is None:
            raise RuntimeError("set_capture_layers() must be called before capture()")
        padded = input_ids.shape[1]
        data, aux_rows, last_rows = [], [], []
        for row, length in enumerate(_real_lengths(attention_mask)):
            row_ids = input_ids[row : row + 1]
            aux, last = self._capture_row(row_ids[:, :length].to(self.output_device))
            aux_rows.append(_pad_rows(aux, padded))
            last_rows.append(_pad_rows(last, padded))
            data.append(
                (
                    row_ids,
                    attention_mask[row : row + 1],
                    loss_mask[row : row + 1],
                )
            )
        return data, aux_rows, last_rows


def _real_lengths(attention_mask: torch.Tensor) -> List[int]:
    """Per-row token counts, refusing anything but right padding."""

    lengths = []
    for row in attention_mask.cpu():
        length = int(row.sum())
        if not bool(row[:length].all()) or bool(row[length:].any()):
            raise ValueError(
                "the HF capture backend expects right-padded rows "
                "(attention_mask = ones then zeros)"
            )
        lengths.append(length)
    return lengths


def _pad_rows(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[0] == length:
        return tensor
    pad = tensor.new_zeros((length - tensor.shape[0], *tensor.shape[1:]))
    return torch.cat([tensor, pad], dim=0)


__all__ = ["OfflineHFCaptureBackend", "dispatch_layers", "plan_layer_devices"]
