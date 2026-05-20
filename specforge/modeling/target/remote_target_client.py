"""Remote target model clients for training/inference separation.

Provides HTTP-based backends that communicate with a standalone
target model server (remote_target_server.py) instead of loading
the model locally.

Transport layers (auto-negotiated via HTTP headers)
---------------------------------------------------
* NCCL (X-SpecForge-Nccl: 1) — GPU→GPU via NCCL send/recv.
  Large tensor data transferred directly GPU-to-GPU; HTTP carries only metadata.
* Custom wire format — compact binary encoding (fallback).

Eagle3
------
Training → POST /generate_eagle3_data
  Request:  input_ids, attention_mask, loss_mask  (torch tensors, serialized)
  Response: hidden_states, target, loss_mask, input_ids  (torch tensors, serialized)

DFlash
------
Training → POST /generate_dflash_data
  Request:  input_ids, attention_mask, loss_mask  (torch tensors, serialized)
  Response: hidden_states, input_ids, attention_mask, loss_mask  (torch tensors, serialized)
"""

import concurrent.futures
import io
import atexit
import itertools
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional
from concurrent.futures import Future

import requests
import torch
import torch.distributed as dist

from . import _tensor_wire as _wire
from ._nccl_transport import NCCLTransport, decode_nccl_metadata
from .dflash_target_model import DFlashTargetModel, DFlashTargetOutput
from .eagle3_target_model import Eagle3TargetModel, Eagle3TargetOutput

logger = logging.getLogger(__name__)


def _get_tp_group_if_distributed():
    """Return the TP process group if distributed training is active with tp>1, else None."""
    if not dist.is_initialized():
        return None
    try:
        from specforge.distributed import get_tp_group
        pg = get_tp_group()
        if pg is not None and dist.get_world_size(pg) > 1:
            return pg
    except (ImportError, RuntimeError):
        pass
    return None


# dtype <-> int mapping for broadcasting tensor metadata
_DTYPE_MAP = {
    torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
    torch.int64: 3, torch.int32: 4, torch.int16: 5, torch.int8: 6,
    torch.uint8: 7, torch.bool: 8, torch.float64: 9,
}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_MAP.items()}


def _dtype_to_int(dtype: torch.dtype) -> int:
    return _DTYPE_MAP.get(dtype, 0)


def _int_to_dtype(i: int) -> torch.dtype:
    return _INT_TO_DTYPE.get(i, torch.float32)


def _tp_broadcast_tensors(tp_group, tp_src, tensors: List[torch.Tensor], flags: List[bool]):
    """Broadcast a list of tensors from tp_src to all ranks in tp_group.

    On rank 0 (sender): packs metadata (flags + per-tensor shape/dtype) and
    broadcasts it, then broadcasts each tensor.

    On other ranks (receiver): receives metadata, allocates buffers, receives
    tensors.

    Returns (received_tensors, flags) on all ranks.
    """
    rank = dist.get_rank(tp_group)
    if rank == 0:
        # Pack metadata: [*flags, num_tensors, then per tensor: ndim, *shape, dtype_int]
        meta = [int(f) for f in flags] + [len(tensors)]
        for t in tensors:
            meta.append(len(t.shape))
            meta.extend(t.shape)
            meta.append(_dtype_to_int(t.dtype))
        meta_t = torch.tensor(meta, dtype=torch.int64, device="cuda")
        len_t = torch.tensor([len(meta)], dtype=torch.int64, device="cuda")
        dist.broadcast(len_t, src=tp_src, group=tp_group)
        dist.broadcast(meta_t, src=tp_src, group=tp_group)
        for t in tensors:
            dist.broadcast(t.contiguous(), src=tp_src, group=tp_group)
        return tensors, [bool(f) for f in flags]
    else:
        len_t = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(len_t, src=tp_src, group=tp_group)
        meta_t = torch.zeros(int(len_t.item()), dtype=torch.int64, device="cuda")
        dist.broadcast(meta_t, src=tp_src, group=tp_group)
        meta = meta_t.tolist()
        num_flags = len(flags)  # caller tells us how many flags to expect
        decoded_flags = [bool(meta[i]) for i in range(num_flags)]
        num_tensors = int(meta[num_flags])
        idx = num_flags + 1
        received = []
        for _ in range(num_tensors):
            ndim = int(meta[idx]); idx += 1
            shape = tuple(int(meta[idx + j]) for j in range(ndim)); idx += ndim
            dtype = _int_to_dtype(int(meta[idx])); idx += 1
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            dist.broadcast(buf, src=tp_src, group=tp_group)
            received.append(buf)
        return received, decoded_flags

# HTTP header name (must match remote_target_server.py)
NCCL_HEADER = "X-SpecForge-Nccl"


# ---------------------------------------------------------------------------
# Helper: serialize / deserialize dicts of tensors
# ---------------------------------------------------------------------------


def _serialize_tensors(data: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a dict of tensors into a bytes buffer via torch.save."""
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def _deserialize_scalar_dict(raw: bytes) -> dict:
    """Deserialize a dict of scalars (returned by /get_model_info)."""
    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Standalone helper: fetch target config from a remote server
# ---------------------------------------------------------------------------


def fetch_remote_target_config(url: str, timeout: int = 120, max_retries: int = 3) -> dict:
    """Lightweight call to /get_model_info that doesn't require a full target model.

    Returns the server's ``model_info`` dict, which includes ``hf_config_dict``
    (the target model's HuggingFace config serialized as a dict) so callers
    can auto-generate draft model configs without local access to the target.
    """
    client = RemoteModelClient(url, timeout, max_retries)
    try:
        raw = client._request("get_model_info", b"")
        return _deserialize_scalar_dict(raw)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Base remote client
# ---------------------------------------------------------------------------


class RemoteModelClient:
    """Shared HTTP client with connection reuse, retry, and backoff."""

    def __init__(
        self,
        url: str,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        # Keep-alive for connection reuse
        self._session.headers.update({"Connection": "keep-alive"})
        # Bypass HTTP_PROXY for localhost connections
        self._session.trust_env = False
        # NCCL transport state
        self._nccl_enabled = os.environ.get("SPECFORGE_ENABLE_NCCL", "1") == "1"
        self._nccl_transport: Optional[NCCLTransport] = None
        self._nccl_init_attempted = False
        self._nccl_init_lock = threading.Lock()
        atexit.register(self.close)

    def _get_nccl_port(self) -> int:
        """Determine the NCCL port for rendezvous."""
        env_port = os.environ.get("SPECFORGE_NCCL_PORT")
        if env_port:
            return int(env_port)
        # Default: HTTP port + 100
        from urllib.parse import urlparse
        parsed = urlparse(self.url)
        http_port = parsed.port or 8001
        return http_port + 100

    def _get_server_host(self) -> str:
        """Get the server host for NCCL TCP rendezvous."""
        from urllib.parse import urlparse
        parsed = urlparse(self.url)
        host = parsed.hostname
        # Loopback aliases connect through 127.0.0.1; otherwise use the URL
        # host so cross-machine clients connect to the advertised server address.
        if host in ("localhost", "::1"):
            return "127.0.0.1"
        return host

    def _init_nccl(self) -> bool:
        """Lazily initialize the NCCL transport (called on first request).

        This sends POST /init_nccl to the server, then both sides block on
        the TCP store rendezvous to establish the NCCL group.

        Returns True on success, False on failure.
        """
        with self._nccl_init_lock:
            if self._nccl_transport is not None and self._nccl_transport.is_initialized:
                return True
            if self._nccl_init_attempted:
                return False  # Already failed once, don't retry

            self._nccl_init_attempted = True
            nccl_port = self._get_nccl_port()
            server_host = self._get_server_host()

            logger.info(
                "Initializing NCCL transport: host=%s, port=%d", server_host, nccl_port
            )

            # Create client-side transport (rank 1)
            self._nccl_transport = NCCLTransport(
                nccl_port=nccl_port,
                host=server_host,
                is_server=False,
            )

            # Start client NCCL init in background thread (blocks on TCP store)
            init_result = [None]
            init_error = [None]

            def _client_init():
                try:
                    init_result[0] = self._nccl_transport.initialize(timeout_seconds=120)
                except Exception as exc:
                    init_error[0] = exc
                    init_result[0] = False

            init_thread = threading.Thread(target=_client_init, daemon=True)
            init_thread.start()

            # Tell the server to also initialize (this triggers server-side rendezvous)
            try:
                resp = self._session.post(
                    f"{self.url}/init_nccl",
                    data=json.dumps({"nccl_port": nccl_port}).encode(),
                    timeout=180,  # generous timeout for NCCL init
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                server_status = json.loads(resp.content.decode())
                if server_status.get("status") not in ("ok", "already_initialized"):
                    logger.warning(
                        "Server NCCL init returned: %s", server_status
                    )
                    self._nccl_transport = None
                    init_thread.join(timeout=5)
                    return False
            except Exception as exc:
                logger.warning("Failed to send /init_nccl to server: %s", exc)
                self._nccl_transport = None
                init_thread.join(timeout=5)
                return False

            # Wait for client-side init to complete
            init_thread.join(timeout=130)
            if init_error[0] is not None:
                logger.warning("Client NCCL init error: %s", init_error[0])
                self._nccl_transport = None
                return False
            if not init_result[0]:
                logger.warning("Client NCCL init failed")
                self._nccl_transport = None
                return False

            logger.info("NCCL transport established successfully")
            return True

    def _request(self, endpoint: str, payload: bytes) -> bytes:
        """POST payload to endpoint with exponential-backoff retry."""
        url = f"{self.url}/{endpoint.lstrip('/')}"
        last_exc = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    url,
                    data=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/octet-stream"},
                )
                resp.raise_for_status()
                return resp.content
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 2**attempt
                    logger.warning(
                        "Remote target request failed (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Remote target request failed after {self.max_retries + 1} attempts: {exc}"
                    ) from last_exc

    def _request_transport(
        self, endpoint: str, payload: bytes, map_location: str = "cpu"
    ) -> dict:
        """Unified transport with auto-negotiation: NCCL > wire format.

        The client signals NCCL capability.  The server picks NCCL
        if the NCCL group is established, otherwise sends compact wire format.
        """
        url = f"{self.url}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/octet-stream"}

        # Lazily initialize NCCL on first request
        if self._nccl_enabled and not self._nccl_init_attempted:
            self._init_nccl()

        # Signal NCCL capability if initialized
        if self._nccl_transport is not None and self._nccl_transport.is_initialized:
            headers[NCCL_HEADER] = "1"

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, data=payload, timeout=self.timeout, headers=headers
                )
                resp.raise_for_status()

                nccl_used = resp.headers.get(NCCL_HEADER) == "1"

                if nccl_used and self._nccl_transport is not None and self._nccl_transport.is_initialized:
                    # NCCL path: HTTP body contains only metadata (JSON)
                    # Tensors arrive via NCCL recv
                    keys_order, metadata, cpu_scalars = decode_nccl_metadata(resp.content)
                    result = self._nccl_transport.recv_tensors(metadata, keys_order)
                    # Reconstruct CPU scalar tensors from JSON
                    for k, v in cpu_scalars.items():
                        result[k] = torch.tensor(v, dtype=torch.int32)
                    return result
                else:
                    return _wire.decode(resp.content, map_location=map_location)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"Remote request failed after {self.max_retries + 1} attempts: {exc}"
                ) from exc

    def close(self):
        if self._nccl_transport is not None:
            self._nccl_transport.destroy()
            self._nccl_transport = None
        self._session.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Remote Eagle3 target model
# ---------------------------------------------------------------------------


class RemoteEagle3TargetModel(Eagle3TargetModel):
    """Eagle3 target model that delegates forward passes to remote servers.

    Supports multiple server URLs for load-balanced inference.
    Setup calls (set_aux_hidden_states_layers, set_vocab_mapping) are
    broadcast to all servers.  Forward calls (generate_eagle3_data) are
    round-robin distributed across the server pool.

    Async prefetch: when enable_prefetch() is called with an iterator,
    the model transparently prefetches the NEXT batch's target data in
    background while the current batch's draft training runs.  This
    overlaps server computation + transfer with client training.
    """

    def __init__(
        self,
        urls: List[str],
        timeout: int = 120,
        max_retries: int = 3,
    ):
        super().__init__()
        self._clients = [RemoteModelClient(u, timeout, max_retries) for u in urls]
        self._next = itertools.cycle(range(len(self._clients)))
        self._executor_cache = None

    @staticmethod
    def _build_eagle3_payload(
        input_ids, attention_mask, loss_mask,
        pixel_values=None, image_grid_thw=None, is_vlm=False,
    ) -> bytes:
        """Build and serialize the Eagle3 request payload."""
        payload = {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "loss_mask": loss_mask.cpu(),
        }
        if is_vlm and pixel_values is not None:
            payload["pixel_values"] = pixel_values.cpu()
            payload["image_grid_thw"] = image_grid_thw.cpu()
            payload["is_vlm"] = torch.tensor(True)
        else:
            payload["is_vlm"] = torch.tensor(is_vlm)
        return _serialize_tensors(payload)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs,
    ) -> "RemoteEagle3TargetModel":
        if urls:
            resolved = urls
        elif url:
            resolved = [url]
        else:
            resolved = [pretrained_model_name_or_path]
        return cls(urls=resolved, timeout=timeout, max_retries=max_retries)

    def _broadcast_setup(self, endpoint: str, payload: bytes) -> None:
        # Only rank 0 sends setup requests when TP > 1
        tp_group = _get_tp_group_if_distributed()
        if tp_group is not None:
            tp_rank = dist.get_rank(tp_group)
            if tp_rank != 0:
                dist.barrier(group=tp_group)
                return
        for client in self._clients:
            client._request(endpoint, payload)
        if tp_group is not None:
            dist.barrier(group=tp_group)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        tp_group = _get_tp_group_if_distributed()
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0

        if aux_hidden_states_layers is None:
            if tp_rank == 0:
                try:
                    info = self._clients[0]._request("get_model_info", b"")
                    info_dict = _deserialize_scalar_dict(info)
                    num_layers = info_dict.get("num_hidden_layers")
                    if num_layers is None:
                        raise ValueError(
                            "Server did not return num_hidden_layers in model info"
                        )
                    aux_hidden_states_layers = [
                        1,
                        num_layers // 2 - 1,
                        num_layers - 4,
                    ]
                except Exception:
                    raise ValueError(
                        "Failed to auto-detect aux_hidden_states_layers from remote server. "
                        "Please pass the layer indices explicitly."
                    )
            # Broadcast the detected layers from rank 0 to other ranks
            if tp_group is not None:
                if tp_rank == 0:
                    layers_t = torch.tensor(aux_hidden_states_layers, dtype=torch.int64, device="cuda")
                else:
                    layers_t = torch.zeros(3, dtype=torch.int64, device="cuda")
                dist.broadcast(layers_t, src=dist.get_global_rank(tp_group, 0), group=tp_group)
                aux_hidden_states_layers = layers_t.tolist()

        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers for EAGLE3"

        layer_payload = _serialize_tensors({"layers": torch.tensor(aux_hidden_states_layers)})
        self._broadcast_setup("set_aux_hidden_states_layers", layer_payload)

    def set_vocab_mapping(self, t2d: torch.Tensor) -> None:
        """Send vocab mapping to all servers for server-side target_p computation."""
        payload = _serialize_tensors({"t2d": t2d.cpu()})
        self._broadcast_setup("set_vocab_mapping", payload)

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
    ) -> Eagle3TargetOutput:
        # If TP > 1, only rank 0 sends the request, then broadcasts to others
        tp_group = _get_tp_group_if_distributed()
        if tp_group is not None:
            return self._generate_eagle3_data_tp(
                tp_group, input_ids, attention_mask, loss_mask,
                pixel_values, image_grid_thw, is_vlm,
            )
        return self._generate_eagle3_data_single(
            input_ids, attention_mask, loss_mask,
            pixel_values, image_grid_thw, is_vlm,
        )

    def _generate_eagle3_data_tp(
        self,
        tp_group,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
    ) -> Eagle3TargetOutput:
        """TP-aware: rank 0 sends request, broadcasts result to other ranks."""
        tp_rank = dist.get_rank(tp_group)
        tp_src = dist.get_global_rank(tp_group, 0)

        if tp_rank == 0:
            output = self._generate_eagle3_data_single(
                input_ids, attention_mask, loss_mask,
                pixel_values, image_grid_thw, is_vlm,
            )
            tensors = [output.hidden_states, output.target, output.loss_mask, output.input_ids]
            flags = [output.last_hidden_states is not None, output.position_mask is not None]
            if flags[0]:
                tensors.append(output.last_hidden_states)
            if flags[1]:
                tensors.append(output.position_mask)
            _tp_broadcast_tensors(tp_group, tp_src, tensors, flags)
            return output
        else:
            received, flags = _tp_broadcast_tensors(tp_group, tp_src, [], [False, False])
            has_last_hidden, has_position_mask = flags
            return Eagle3TargetOutput(
                hidden_states=received[0],
                target=received[1],
                loss_mask=received[2],
                input_ids=received[3],
                attention_mask=attention_mask,
                last_hidden_states=received[4] if has_last_hidden else None,
                position_mask=received[5 if has_last_hidden else 4] if has_position_mask else None,
            )

    def _result_to_eagle3_output(self, result: dict, attention_mask: torch.Tensor) -> Eagle3TargetOutput:
        """Convert a transport result dict to Eagle3TargetOutput on CUDA."""
        if "target_topk_vals" in result:
            topk_vals = result["target_topk_vals"].cuda().float()
            topk_indices = result["target_topk_indices"].cuda().long()
            vocab_size = result["target_vocab_size"].item()
            batch_shape = topk_vals.shape[:-1]
            target_p = torch.zeros(*batch_shape, vocab_size, device=topk_vals.device, dtype=topk_vals.dtype)
            target_p.scatter_(-1, topk_indices, topk_vals)
            target_tensor = target_p.to(topk_vals.dtype)
        else:
            target_tensor = result["target"].cuda()
        return Eagle3TargetOutput(
            hidden_states=result["hidden_states"].cuda(),
            target=target_tensor,
            loss_mask=result["loss_mask"].cuda(),
            input_ids=result["input_ids"].cuda(),
            attention_mask=attention_mask,
            last_hidden_states=(
                result["last_hidden_states"].cuda()
                if "last_hidden_states" in result and result["last_hidden_states"] is not None
                else None
            ),
            position_mask=(
                result["position_mask"].cuda()
                if "position_mask" in result and result["position_mask"] is not None
                else None
            ),
        )

    def _generate_eagle3_data_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
    ) -> Eagle3TargetOutput:
        payload_bytes = self._build_eagle3_payload(
            input_ids, attention_mask, loss_mask,
            pixel_values, image_grid_thw, is_vlm,
        )

        client = self._clients[next(self._next)]
        result = client._request_transport(
            "generate_eagle3_data", payload_bytes
        )
        return self._result_to_eagle3_output(result, attention_mask)

    def generate_eagle3_data_async(
        self,
        input_ids,
        attention_mask,
        loss_mask,
        pixel_values=None,
        image_grid_thw=None,
        is_vlm=False,
    ) -> Future:
        """Submit an async forward pass.  Returns a ``Future[Eagle3TargetOutput]``.

        Useful with ``--draft-accumulation-steps > 1`` to pipeline the
        forward passes across the server pool.
        """
        client = self._clients[next(self._next)]
        # Pre-serialise payload so the background thread doesn't hold the
        # GIL across PyTorch ops in the main thread.
        payload_bytes = self._build_eagle3_payload(
            input_ids, attention_mask, loss_mask,
            pixel_values, image_grid_thw, is_vlm,
        )

        def _do_request():
            result = client._request_transport("generate_eagle3_data", payload_bytes)
            return self._result_to_eagle3_output(result, attention_mask)

        return self._executor.submit(_do_request)

    @property
    def _executor(self):
        if self._executor_cache is None:
            self._executor_cache = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self._clients)
            )
        return self._executor_cache

    def close(self):
        if self._executor_cache is not None:
            self._executor_cache.shutdown(wait=False, cancel_futures=True)
            self._executor_cache = None
        for client in self._clients:
            client.close()


class RemoteDFlashTargetModel(DFlashTargetModel):
    """DFlash target model that delegates forward passes to remote servers.

    Supports multiple server URLs for load-balanced inference.
    Setup calls are broadcast to all servers.  Forward calls are round-robin
    distributed across the server pool.
    """

    def __init__(
        self,
        urls: List[str],
        timeout: int = 120,
        max_retries: int = 3,
    ):
        super().__init__()
        self._clients = [RemoteModelClient(u, timeout, max_retries) for u in urls]
        self._next = itertools.cycle(range(len(self._clients)))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs,
    ) -> "RemoteDFlashTargetModel":
        if urls:
            resolved = urls
        elif url:
            resolved = [url]
        else:
            resolved = [pretrained_model_name_or_path]
        return cls(urls=resolved, timeout=timeout, max_retries=max_retries)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)
        layer_payload = _serialize_tensors({"layers": torch.tensor(layer_ids)})
        # Only rank 0 sends when TP > 1
        tp_group = _get_tp_group_if_distributed()
        if tp_group is not None:
            tp_rank = dist.get_rank(tp_group)
            if tp_rank != 0:
                dist.barrier(group=tp_group)
                return
        for client in self._clients:
            client._request("set_capture_layers", layer_payload)
        if tp_group is not None:
            dist.barrier(group=tp_group)

    @staticmethod
    def _result_to_dflash_output(result: dict) -> DFlashTargetOutput:
        """Convert a transport result dict to DFlashTargetOutput on CUDA."""
        return DFlashTargetOutput(
            hidden_states=result["hidden_states"].cuda(),
            input_ids=result["input_ids"].cuda(),
            attention_mask=result["attention_mask"].cuda(),
            loss_mask=result["loss_mask"].cuda(),
            position_ids=(
                result["position_ids"].cuda()
                if "position_ids" in result and result["position_ids"] is not None
                else None
            ),
        )

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> DFlashTargetOutput:
        if pixel_values is not None or image_grid_thw is not None or video_grid_thw is not None:
            raise NotImplementedError(
                "Remote DFlash target does not yet support multimodal inputs."
            )
        # If TP > 1, only rank 0 sends the request, then broadcasts to others
        tp_group = _get_tp_group_if_distributed()
        if tp_group is not None:
            return self._generate_dflash_data_tp(
                tp_group, input_ids, attention_mask, loss_mask,
            )
        payload = {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "loss_mask": loss_mask.cpu(),
        }
        # Round-robin across server pool
        client = self._clients[next(self._next)]
        result = client._request_transport(
            "generate_dflash_data", _serialize_tensors(payload)
        )
        return self._result_to_dflash_output(result)

    def _generate_dflash_data_tp(
        self,
        tp_group,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        """TP-aware: rank 0 sends request, broadcasts result to other ranks."""
        tp_rank = dist.get_rank(tp_group)
        tp_src = dist.get_global_rank(tp_group, 0)

        if tp_rank == 0:
            payload = {
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
                "loss_mask": loss_mask.cpu(),
            }
            client = self._clients[next(self._next)]
            result = client._request_transport(
                "generate_dflash_data", _serialize_tensors(payload)
            )
            output = self._result_to_dflash_output(result)

            tensors = [output.hidden_states, output.input_ids, output.attention_mask, output.loss_mask]
            flags = [output.position_ids is not None]
            if flags[0]:
                tensors.append(output.position_ids)
            _tp_broadcast_tensors(tp_group, tp_src, tensors, flags)
            return output
        else:
            received, flags = _tp_broadcast_tensors(tp_group, tp_src, [], [False])
            has_position_ids = flags[0]
            return DFlashTargetOutput(
                hidden_states=received[0],
                input_ids=received[1],
                attention_mask=received[2],
                loss_mask=received[3],
                position_ids=received[4] if has_position_ids else None,
            )

    def close(self):
        for client in self._clients:
            client.close()
