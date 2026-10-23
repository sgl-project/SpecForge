"""Offline capture from a running vLLM / vLLM-Ascend hidden-state server.

The server runs vLLM's ``extract_hidden_states`` speculative method with
``ExampleHiddenStatesConnector``: each ``/v1/completions`` request with
``max_tokens=1`` is one prefill, and the connector writes that prompt's
auxiliary hidden states to ``{shared_storage_path}/{req_id}.safetensors``
(``hidden_states [T, K, H]`` and ``token_ids [T]``), returning the path in
``kv_transfer_params.hidden_states_path``.

The server's layer set is fixed when it starts. For DFlash-family features it
must be the capture layers in vLLM's numbering -- SpecForge's id ``i`` is the
output of layer ``i``, vLLM's aux id ``i + 1`` is the input of that next layer
-- followed by ``num_hidden_layers`` for the state ``lm_head`` reads::

    eagle_aux_hidden_state_layer_ids = [i + 1 for i in capture_layers] + [L]

The connector stores them in ascending order, so slots ``0..K-2`` are the
capture layers in order and slot ``K-1`` is the final state. This backend
splits them into the same ``(aux, last)`` rows the other backends return.

Prompts are sent as token ids, never as text: the loss mask comes from
SpecForge's own rendering, so the server must not render a chat template.

The connector must keep ``use_synchronization_lock`` on (its default); see
:func:`load_capture_file`. Serving Qwen3.8-Flash-Next with
``extract_hidden_states`` needs changes not yet in upstream vLLM or
vLLM-Ascend; ``docs/sections/basic_usage/data_preparation.md`` names them.
"""

from __future__ import annotations

import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch


def _num_hidden_layers(model_path: str) -> int:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    text = config.get("text_config") or config
    return int(text["num_hidden_layers"])


def parse_path_map(spec: Optional[str]) -> Optional[Tuple[str, str]]:
    """``SERVER_PREFIX=LOCAL_PREFIX`` -> the pair, for servers in another container."""

    if not spec:
        return None
    server, sep, local = spec.partition("=")
    if not sep or not server or not local:
        raise ValueError(
            f"--vllm-path-map must look like SERVER_PREFIX=LOCAL_PREFIX, got {spec!r}"
        )
    return server, local


def load_capture_file(path: str) -> Dict[str, torch.Tensor]:
    """Read a connector file, waiting on its lock rather than polling.

    The connector writes asynchronously and holds an exclusive ``flock`` on
    ``<path>.lock`` until the file is complete; a shared lock sleeps until then.
    The write is asynchronous with or without that lock, so the lock is the
    only signal that the file is complete: a server started with
    ``use_synchronization_lock=False`` is refused rather than raced.
    """

    from safetensors.torch import load_file

    lock_path = path + ".lock"
    if not os.path.exists(lock_path):
        raise RuntimeError(
            f"capture file {path} has no {os.path.basename(lock_path)}: the "
            "connector writes asynchronously and the lock is the only sign the "
            "file is complete. Start the server with use_synchronization_lock "
            "enabled (the ExampleHiddenStatesConnector default) in "
            "kv_connector_extra_config."
        )
    with open(lock_path) as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        return load_file(path, device="cpu")


class OfflineVLLMCaptureBackend:
    """Remote vLLM target whose rows come back unpadded."""

    # Rows are captured from their real tokens only; the preparation script
    # trims each saved record to its attention-mask length.
    unpadded_rows = True

    def __init__(
        self,
        *,
        url: str,
        served_model_name: str,
        num_hidden_layers: int,
        concurrency: int = 4,
        timeout: float = 600.0,
        keep_captures: bool = False,
        path_map: Optional[Tuple[str, str]] = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError(f"--vllm-concurrency must be >= 1, got {concurrency}")
        self.url = url.rstrip("/")
        self.served_model_name = served_model_name
        self.num_hidden_layers = int(num_hidden_layers)
        self.concurrency = int(concurrency)
        self.timeout = float(timeout)
        self.keep_captures = bool(keep_captures)
        self.path_map = path_map
        self.layer_ids: Optional[List[int]] = None
        self._pool = ThreadPoolExecutor(max_workers=self.concurrency)

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        *,
        url: str,
        served_model_name: Optional[str] = None,
        concurrency: int = 4,
        timeout: float = 600.0,
        keep_captures: bool = False,
        path_map: Optional[str] = None,
        **_unused,
    ) -> "OfflineVLLMCaptureBackend":
        return cls(
            url=url,
            served_model_name=served_model_name or pretrained_model_name_or_path,
            num_hidden_layers=_num_hidden_layers(pretrained_model_name_or_path),
            concurrency=concurrency,
            timeout=timeout,
            keep_captures=keep_captures,
            path_map=parse_path_map(path_map),
        )

    def server_layer_ids(self) -> List[int]:
        """The ``eagle_aux_hidden_state_layer_ids`` the server must run with."""

        if self.layer_ids is None:
            raise RuntimeError("set_capture_layers() must be called first")
        return [i + 1 for i in self.layer_ids] + [self.num_hidden_layers]

    def set_capture_layers(
        self,
        layer_ids: Optional[List[int]] = None,
        *,
        capture_method: str,
    ) -> None:
        if capture_method not in {"dflash", "dspark"}:
            raise ValueError(
                "the vLLM capture backend writes DFlash-family features only, "
                f"got capture method {capture_method!r}"
            )
        if not layer_ids:
            raise ValueError("the vLLM capture backend needs explicit capture layers")
        if list(layer_ids) != sorted(set(layer_ids)):
            raise ValueError(
                "capture layers must be unique and ascending, because the server "
                f"returns its layers in ascending order; got {list(layer_ids)}"
            )
        if layer_ids[-1] >= self.num_hidden_layers - 1:
            raise ValueError(
                f"capture layer {layer_ids[-1]} has no following layer in a "
                f"{self.num_hidden_layers}-layer target"
            )
        self.layer_ids = list(layer_ids)
        print(
            "vLLM capture: the server must run with eagle_aux_hidden_state_layer_ids="
            f"{self.server_layer_ids()}",
            flush=True,
        )

    def _local_path(self, path: str) -> str:
        if self.path_map is None:
            return path
        server, local = self.path_map
        if not path.startswith(server):
            raise ValueError(
                f"server path {path!r} is outside --vllm-path-map prefix {server!r}"
            )
        return local + path[len(server) :]

    def _request(self, token_ids: List[int]) -> str:
        import requests

        response = requests.post(
            f"{self.url}/v1/completions",
            json={
                "model": self.served_model_name,
                "prompt": token_ids,
                "max_tokens": 1,
                "return_token_ids": True,
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"capture request failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        payload = response.json()
        echoed = payload["choices"][0].get("prompt_token_ids")
        if echoed is None or list(echoed) != token_ids:
            raise RuntimeError(
                "the server's prompt_token_ids differ from the ids sent; it must "
                "take token ids verbatim, without rendering a chat template"
            )
        path = (payload.get("kv_transfer_params") or {}).get("hidden_states_path")
        if not path:
            raise RuntimeError(
                "the response has no kv_transfer_params.hidden_states_path; is the "
                "server running extract_hidden_states with ExampleHiddenStatesConnector?"
            )
        return self._local_path(path)

    def capture_ids(self, token_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """One prompt's ``(aux [T, n*H], last [T, H])``, checked against what was sent."""

        if self.layer_ids is None:
            raise RuntimeError("set_capture_layers() must be called before capture()")
        path = self._request(token_ids)
        try:
            data = load_capture_file(path)
        finally:
            if not self.keep_captures:
                for leftover in (path, path + ".lock"):
                    if os.path.exists(leftover):
                        os.remove(leftover)

        hidden, saved_ids = data["hidden_states"], data["token_ids"]
        if saved_ids.tolist() != token_ids:
            raise RuntimeError("the capture file's token_ids differ from the ids sent")
        expected = len(self.layer_ids) + 1
        if hidden.dim() != 3 or hidden.shape[0] != len(token_ids):
            raise RuntimeError(
                f"expected hidden_states [{len(token_ids)}, {expected}, H], got "
                f"{tuple(hidden.shape)}"
            )
        if hidden.shape[1] != expected:
            raise RuntimeError(
                f"the server returned {hidden.shape[1]} layers per token, but "
                f"{len(self.layer_ids)} capture layers plus the final state need "
                f"{expected}: start it with eagle_aux_hidden_state_layer_ids="
                f"{self.server_layer_ids()}"
            )
        if not torch.isfinite(hidden.float()).all():
            raise RuntimeError("the captured hidden_states contain NaN or Inf")
        length, _, width = hidden.shape
        aux = hidden[:, :-1, :].reshape(length, (expected - 1) * width)
        return aux, hidden[:, -1, :]

    @torch.no_grad()
    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Capture a batch, one concurrent request per row.

        Each row is sent without its right padding; its states are zero-padded
        back to the batch length so the rows stack, and the preparation script
        trims every saved record to its real length (``unpadded_rows``).
        """

        lengths = _real_lengths(attention_mask)
        rows = [input_ids[row, :length].tolist() for row, length in enumerate(lengths)]
        results = list(self._pool.map(self.capture_ids, rows))

        padded = input_ids.shape[1]
        data, aux_rows, last_rows = [], [], []
        for row, (aux, last) in enumerate(results):
            aux_rows.append(_pad_rows(aux, padded))
            last_rows.append(_pad_rows(last, padded))
            data.append(
                (
                    input_ids[row : row + 1].cpu(),
                    attention_mask[row : row + 1].cpu(),
                    loss_mask[row : row + 1].cpu(),
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
                "the vLLM capture backend expects right-padded rows "
                "(attention_mask = ones then zeros)"
            )
        lengths.append(length)
    return lengths


def _pad_rows(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[0] == length:
        return tensor
    pad = tensor.new_zeros((length - tensor.shape[0], *tensor.shape[1:]))
    return torch.cat([tensor, pad], dim=0)


__all__ = [
    "OfflineVLLMCaptureBackend",
    "load_capture_file",
    "parse_path_map",
]
