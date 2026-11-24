"""Producer-side wire codec for the TPU capture server (specforge_tpu.sglang_capture).

sglang-torchtpu's GenerateReqInput has no spec_capture field, so each request's
spec rides in sampling_params.custom_params as a JSON string.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from typing import Any

SPEC_KEY = "spec_capture"


def encode_capture_body(body: dict[str, Any]) -> dict[str, Any]:
    specs = body.pop(SPEC_KEY, None)
    if specs is None:
        return body
    shared = dict(body.get("sampling_params") or {})
    body["sampling_params"] = [
        {**shared, "custom_params": {SPEC_KEY: json.dumps(spec)}} for spec in specs
    ]
    body["return_hidden_states"] = True
    return body


def wrap_post_fn(post_fn: Callable[..., Any]) -> Callable[..., Any]:
    def post(url: str, *, json_body: dict[str, Any], **kwargs: Any) -> Any:
        return post_fn(url, json_body=encode_capture_body(json_body), **kwargs)

    return post


def wrap_adapter_init(original: Callable[..., None]) -> Callable[..., None]:
    @wraps(original)
    def init(adapter: Any, *args: Any, **kwargs: Any) -> None:
        original(adapter, *args, **kwargs)
        adapter.post_fn = wrap_post_fn(adapter.post_fn)

    return init
