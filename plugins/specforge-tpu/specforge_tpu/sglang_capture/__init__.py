"""SpecForge spec capture for sglang-torchtpu, as an sglang general plugin.

The TPU counterpart of patches/sglang/v0.5.18/spec-capture.patch, applied
through sglang's HookRegistry instead of a source patch. Enabled by
SPECFORGE_SPEC_CAPTURE_LAYER_IDS (the draft's target_layer_ids).

Wire contract (see specforge_tpu.producer): the producer sends
return_hidden_states=True and the per-request capture spec as a JSON string in
sampling_params.custom_params["spec_capture"]; the result comes back as
meta_info["spec_capture"] = [result].

The logits processor appends the post-norm final hidden to the packed aux
capture, so one [T, (n + 1) * H] buffer carries both the aux features and
target_last_hidden_states.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

LAYER_IDS_ENV = "SPECFORGE_SPEC_CAPTURE_LAYER_IDS"
SPEC_KEY = "spec_capture"

_aux_layer_ids: list[int] = []
_sink: Any = None


def _layer_ids() -> list[int]:
    return [int(item) for item in os.environ[LAYER_IDS_ENV].split(",")]


def enable_target_capture(result: Any, *args: Any, model: Any, **kwargs: Any) -> Any:
    """AFTER configure_aux_hidden_state_capture: add the capture layers."""
    if kwargs.get("dflash_use_aux_hidden_state") or kwargs.get(
        "eagle_use_aux_hidden_state"
    ):
        raise ValueError("spec capture cannot share a server with speculative decoding")
    model.set_dflash_layers_to_capture(_aux_layer_ids)
    return result


def store_last_hidden(
    original: Callable[..., Any],
    processor: Any,
    hidden_states: torch.Tensor,
    hidden_states_before_norm: torch.Tensor | None,
    aux_hidden_states: Any,
    *args: Any,
) -> Any:
    """AROUND LogitsProcessor._get_hidden_states_to_store: append post-norm hidden.

    The layer-boundary capture of the last layer is the pre-norm residual;
    the training target is the post-norm hidden fed to the LM head.
    """
    stored = original(
        processor, hidden_states, hidden_states_before_norm, aux_hidden_states, *args
    )
    logits_metadata = args[-1]
    if aux_hidden_states is None or not logits_metadata.capture_hidden_mode.is_full():
        return stored
    if hidden_states_before_norm is not None:
        raise RuntimeError(
            "spec capture does not support models returning pre-norm hidden"
        )
    return torch.cat([stored, hidden_states.to(stored.dtype)], dim=-1)


def _request_spec(req: Any) -> dict[str, Any] | None:
    cached = req.__dict__.get("_specforge_spec")
    if cached is not None:
        return cached
    custom = req.sampling_params.custom_params
    if not isinstance(custom, dict) or SPEC_KEY not in custom:
        return None
    spec = json.loads(custom[SPEC_KEY])
    req._specforge_spec = spec
    req._specforge_rows = []
    return spec


def _is_writer() -> bool:
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().tp_rank == 0


def append_prefill_hidden_states(
    original: Callable[..., int],
    processor: Any,
    *,
    req: Any,
    logits_output: Any,
    hidden_state_offset: int,
    capture_hidden_mode: Any,
    extend_input_len: int,
    store: bool = True,
) -> int:
    """AROUND _append_prefill_hidden_states: route capture rows to Mooncake.

    Rows of every prefill chunk are kept, so chunked prefill is supported.
    """
    spec = _request_spec(req)
    if spec is None:
        return original(
            processor,
            req=req,
            logits_output=logits_output,
            hidden_state_offset=hidden_state_offset,
            capture_hidden_mode=capture_hidden_mode,
            extend_input_len=extend_input_len,
            store=store,
        )
    if not capture_hidden_mode.is_full():
        raise RuntimeError(
            f"spec capture needs FULL hidden capture, got {capture_hidden_mode}"
        )
    end = hidden_state_offset + extend_input_len
    if req.finished() or req.is_retracted or not _is_writer():
        return end
    rows = logits_output.hidden_states[hidden_state_offset:end]
    req._specforge_rows.append(rows.detach().to("cpu", copy=True))
    if not store or req.inflight_middle_chunks > 0:
        return end

    rows = torch.cat(req._specforge_rows, dim=0)
    req._specforge_rows = []
    split = len(_aux_layer_ids) * (rows.shape[1] // (len(_aux_layer_ids) + 1))
    try:
        if rows.shape[0] != len(req.origin_input_ids):
            raise RuntimeError(
                f"captured {rows.shape[0]} rows for a {len(req.origin_input_ids)}-token prompt"
            )
        (result,) = _sink.put_samples([(spec, rows[:, :split], rows[:, split:])])
    except Exception as error:
        logger.exception("spec capture failed for %s", spec.get("sample_id"))
        result = {"sample_id": spec.get("sample_id"), "error": repr(error)}
    req._specforge_result = result
    return end


def stream_capture_result(result: Any, accumulator: Any, *, req: Any) -> Any:
    """AFTER _GenerationStreamAccumulator.accept: attach the capture result.

    req.customized_info is sliced per output token, which drops it for a
    max_new_tokens=0 capture request, so the row is written here instead.
    """
    capture = req.__dict__.get("_specforge_result")
    if capture is None or not accumulator.rids or accumulator.rids[-1] != req.rid:
        return result
    values = accumulator.customized_info.setdefault(SPEC_KEY, [])
    values.extend([] for _ in range(len(accumulator.rids) - 1 - len(values)))
    if len(values) == len(accumulator.rids):
        values[-1] = [capture]
    else:
        values.append([capture])
    return result


def register() -> None:
    global _aux_layer_ids, _sink
    if not os.environ.get(LAYER_IDS_ENV):
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from specforge_tpu.sglang_capture.sink import CaptureSink

    _aux_layer_ids = _layer_ids()
    _sink = CaptureSink(_aux_layer_ids)
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.configure_aux_hidden_state_capture",
        enable_target_capture,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.layers.logits_processor.LogitsProcessor._get_hidden_states_to_store",
        store_last_hidden,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler_components.batch_result_processor."
        "SchedulerBatchResultProcessor._append_prefill_hidden_states",
        append_prefill_hidden_states,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler_components.output_streamer."
        "_GenerationStreamAccumulator.accept",
        stream_capture_result,
        HookType.AFTER,
    )
    logger.info("SpecForge spec capture enabled for layers %s", _aux_layer_ids)
