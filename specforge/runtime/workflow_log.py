"""Env-gated tracing of the disaggregated training workflow, for studying it.

Set ``SPECFORGE_WORKFLOW_LOG=1`` to print one line per pipeline event, tagged
with the process role and the source location that emitted it::

    [workflow] producer   server_capture.py:342  POST /generate url=... n_prompts=8
    [workflow] trainer    mooncake_store.py:601  fetched sample ... hidden_states=Tensor(1, 812, 7680):bfloat16

Silent (single boolean check) when the variable is unset, so instrumentation
can stay in place permanently.
"""

from __future__ import annotations

import os
import sys

ENABLED = os.environ.get("SPECFORGE_WORKFLOW_LOG", "0") == "1"


def _fmt(v) -> str:
    """Compact one-line rendering; tensors become shape:dtype summaries."""
    try:
        import torch

        if isinstance(v, torch.Tensor):
            dtype = str(v.dtype).replace("torch.", "")
            return f"Tensor{tuple(v.shape)}:{dtype}@{v.device}"
    except Exception:  # pragma: no cover - torch absent in some processes
        pass
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        if len(v) > 6:
            head = ", ".join(_fmt(x) for x in v[:6])
            return f"[{head}, ... +{len(v) - 6} more]"
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def wlog(role: str, event: str, **fields) -> None:
    """Print one workflow event. ``role`` names the process (producer, trainer,
    launcher, capture-server); caller file:line is added automatically."""
    if not ENABLED:
        return
    frame = sys._getframe(1)
    where = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
    kv = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    print(f"[workflow] {role:<10s} {where:<28s} {event} {kv}".rstrip(), flush=True)
