"""HTTP server that wraps SGLang target models for remote training.

Launched via scripts/launch_target_server.py, this module:
  1. Loads an SGLangRunner + SGLang*TargetModel (existing code)
  2. Exposes HTTP endpoints so training scripts on other GPUs/machines
     can call generate_eagle3_data() / generate_dflash_data() remotely.

Transport backends (auto-selected via HTTP headers)
----------------------------------------------------
* NCCL (X-SpecForge-Nccl: 1) — GPU→GPU via NCCL send/recv.
* Custom wire format — compact binary encoding (fallback).

Endpoints
---------
POST /generate_eagle3_data   – Eagle3 hidden states + logits
POST /generate_dflash_data   – DFlash hidden states
POST /init_nccl              – Initialize NCCL data transfer group
POST /set_aux_hidden_states_layers – Configure Eagle3 layer capture
POST /set_capture_layers      – Configure DFlash layer capture
POST /set_vocab_mapping       – Receive t2d vocab mapping
POST /get_model_info          – Return model metadata (num layers, etc.)
POST /disconnect              – Client disconnect notification
POST /heartbeat               – Client heartbeat ping
GET  /health                  – Liveness check

Client Lifecycle
----------------
The server tracks client connectivity via:
  1. Explicit /disconnect notification (sent by client on close)
  2. Heartbeat timeout (detects abnormal client termination)

On client disconnect, the server shuts down automatically.
"""

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import torch
import torch.distributed as dist

from . import _tensor_wire as _wire
from ._nccl_transport import NCCLTransport, encode_nccl_metadata
from .dflash_target_model import SGLangDFlashTargetModel
from .eagle3_target_model import SGLangEagle3TargetModel

logger = logging.getLogger(__name__)

# HTTP header names
NCCL_HEADER = "X-SpecForge-Nccl"

# Endpoint groups used by request prelude.
_LIFECYCLE_ENDPOINTS = {"/disconnect", "/heartbeat"}
_CLIENT_ACTIVITY_ENDPOINTS = {
    "/generate_eagle3_data",
    "/generate_dflash_data",
    "/init_nccl",
    "/set_aux_hidden_states_layers",
    "/set_capture_layers",
    "/get_model_info",
    "/set_vocab_mapping",
    "/heartbeat",
}

# ---------------------------------------------------------------------------
# Wire-format serialization
# ---------------------------------------------------------------------------


def _deserialize_tensors(raw: bytes, map_location: str = "cpu") -> dict:
    """Deserialize REQUEST payloads (tensor dicts) from wire format."""
    return _wire.decode(raw, map_location=map_location)


# ---------------------------------------------------------------------------
# TargetModelServer – wraps one backend and serves it over HTTP
# ---------------------------------------------------------------------------


# Reuse baseline's compiled _compute_target_p to guarantee identical results.
from specforge.utils import padding


class TargetModelServer:
    """Manages a single target model and its lifecycle."""

    def __init__(
        self,
        mode: str,
        model_path: str,
        tp_size: int = 1,
        mem_fraction_static: float = 0.85,
        trust_remote_code: bool = False,
        enable_torch_compile: bool = True,
        nccl_port: int = None,
        host: str = "0.0.0.0",
        attention_backend: str | None = None,
        client_heartbeat_timeout: float = 60.0,
    ):
        self.mode = mode
        self.model_path = model_path
        self.tp_size = tp_size
        self.mem_fraction_static = mem_fraction_static
        self.trust_remote_code = trust_remote_code
        self.enable_torch_compile = enable_torch_compile
        self.attention_backend = attention_backend
        self._model = None
        self._vocab_t2d: torch.Tensor = None  # set via /set_vocab_mapping
        self._vocab_t2d_cuda: torch.Tensor = None  # cached CUDA version
        self._gpu_id: int = None  # set after model loading
        self._pending_nccl_send = None  # stashed for deferred NCCL send
        # NCCL transport for GPU-to-GPU data transfer
        self._nccl_port = nccl_port
        self._host = host
        self._nccl_transport: NCCLTransport = None
        self._nccl_enabled = os.environ.get("SPECFORGE_ENABLE_NCCL", "1") == "1"
        # Cache env vars read on every forward pass
        self._topk = int(os.environ.get("SPECFORGE_TOPK", "0"))
        _target_dtype = os.environ.get("SPECFORGE_TARGET_DTYPE", "fp32").lower()
        self._use_bf16_target = _target_dtype != "fp32"
        # Client disconnect handling
        self._client_heartbeat_timeout = client_heartbeat_timeout
        self._last_client_activity: float = 0.0  # 0 means no client connected yet
        self._client_connected = False
        self._heartbeat_timer: threading.Timer = None
        self._client_lifecycle_lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_callback = None  # set by launcher to trigger server shutdown

    def load_model(self):
        """Load the target model (reuses existing SGLang*TargetModel classes)."""
        kwargs = dict(
            mem_fraction_static=self.mem_fraction_static,
            enable_torch_compile=self.enable_torch_compile,
            attention_backend=self.attention_backend,
        )
        if self.mode == "eagle3":
            self._model = SGLangEagle3TargetModel.from_pretrained(
                pretrained_model_name_or_path=self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=self.trust_remote_code,
                **kwargs,
            )
            logger.info("Loaded SGLangEagle3TargetModel for %s", self.model_path)
        elif self.mode == "dflash":
            kwargs["enforce_disable_flashinfer_allreduce_fusion"] = True
            self._model = SGLangDFlashTargetModel.from_pretrained(
                pretrained_model_name_or_path=self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=self.trust_remote_code,
                **kwargs,
            )
            logger.info("Loaded SGLangDFlashTargetModel for %s", self.model_path)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Store the CUDA device so worker threads can set_device before GPU ops.
        # ThreadingHTTPServer spawns threads that default to cuda:0; the model
        # may live on a different device (e.g. cuda:1 via SPECFORGE_GPU_ID).
        self._gpu_id = torch.cuda.current_device()

        # Pre-set defaults for layers so model is ready
        if self.mode == "eagle3":
            self._model.set_aux_hidden_states_layers()

    def close(self):
        self._stop_heartbeat_timer()
        if self._nccl_transport is not None:
            self._nccl_transport.destroy()
            self._nccl_transport = None

    # ------------------------------------------------------------------
    # Client lifecycle / heartbeat
    # ------------------------------------------------------------------

    def _record_client_activity(self):
        """Record that the client is alive (called on client requests)."""
        with self._client_lifecycle_lock:
            if self._shutdown_requested:
                return
            self._last_client_activity = time.time()
            if self._client_connected:
                return
            self._client_connected = True
            logger.info("Client connected (first activity detected)")
            self._start_heartbeat_timer_locked()

    def _start_heartbeat_timer_locked(self):
        """Start the heartbeat watchdog timer. Caller must hold lifecycle lock."""
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.cancel()
            self._heartbeat_timer = None
        if self._client_heartbeat_timeout <= 0:
            return
        self._heartbeat_timer = threading.Timer(
            self._client_heartbeat_timeout, self._check_heartbeat
        )
        self._heartbeat_timer.daemon = True
        self._heartbeat_timer.start()

    def _stop_heartbeat_timer(self):
        """Cancel the heartbeat watchdog timer."""
        with self._client_lifecycle_lock:
            self._stop_heartbeat_timer_locked()

    def _stop_heartbeat_timer_locked(self):
        """Cancel the heartbeat watchdog timer. Caller must hold lifecycle lock."""
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.cancel()
            self._heartbeat_timer = None

    def _check_heartbeat(self):
        """Called when heartbeat timer fires. Check if client is still alive."""
        should_shutdown = False
        with self._client_lifecycle_lock:
            self._heartbeat_timer = None
            if not self._client_connected or self._client_heartbeat_timeout <= 0:
                return
            elapsed = time.time() - self._last_client_activity
            if elapsed >= self._client_heartbeat_timeout:
                logger.warning(
                    "Client heartbeat timeout (%.1fs since last activity, threshold=%.1fs). "
                    "Treating as disconnected.",
                    elapsed,
                    self._client_heartbeat_timeout,
                )
                self._client_connected = False
                should_shutdown = True
            else:
                remaining = self._client_heartbeat_timeout - elapsed
                self._heartbeat_timer = threading.Timer(
                    remaining + 1.0, self._check_heartbeat
                )
                self._heartbeat_timer.daemon = True
                self._heartbeat_timer.start()
        if should_shutdown:
            self._request_shutdown(reason="heartbeat_timeout")

    def _handle_client_disconnect(self, reason: str = "explicit"):
        """Handle client disconnection by shutting down the server."""
        with self._client_lifecycle_lock:
            self._client_connected = False
            self._stop_heartbeat_timer_locked()
        self._request_shutdown(reason=reason)

    def _request_shutdown(self, reason: str):
        """Request server shutdown exactly once."""
        with self._client_lifecycle_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
        logger.info("Client disconnected (reason=%s). Shutting down server...", reason)
        if self._shutdown_callback is not None:
            self._shutdown_callback()

    def handle_disconnect(self, _raw_body: bytes) -> bytes:
        """Handle explicit client disconnect notification (POST /disconnect)."""
        self._handle_client_disconnect(reason="explicit")
        return json.dumps({"status": "ok"}).encode()

    def handle_heartbeat(self, _raw_body: bytes) -> bytes:
        """Handle client heartbeat ping (POST /heartbeat)."""
        return json.dumps({"status": "ok"}).encode()

    @property
    def _keep_on_gpu(self) -> bool:
        """Whether tensors should stay on GPU (NCCL path) or move to CPU."""
        return (
            getattr(_request_local, "use_nccl", False)
            and self._nccl_transport is not None
            and self._nccl_transport.is_initialized
        )

    def _serialize_response(self, data: dict) -> bytes:
        """Serialise handler response dict to bytes.

        Priority: NCCL (GPU→GPU send) > wire format.

        For NCCL: returns metadata only (no send here — send happens AFTER
        the HTTP response is written, to avoid deadlock with client recv).
        """
        if self._keep_on_gpu:
            # NCCL: encode metadata only.  The actual send is deferred until
            # after the HTTP response is flushed (see _send_response flow).
            torch.cuda.synchronize()
            # Ensure all tensors are contiguous
            for k in list(data.keys()):
                t = data[k]
                if (
                    t is not None
                    and hasattr(t, "is_cuda")
                    and t.is_cuda
                    and not t.is_contiguous()
                ):
                    data[k] = t.contiguous()
            keys_order = [
                k
                for k in data.keys()
                if data[k] is not None
                and hasattr(data[k], "is_cuda")
                and data[k].is_cuda
            ]
            # Collect CPU scalar tensors to include in metadata JSON
            cpu_scalars = {}
            for k, v in data.items():
                if (
                    v is not None
                    and isinstance(v, torch.Tensor)
                    and not v.is_cuda
                    and v.numel() <= 8
                ):
                    cpu_scalars[k] = v.tolist()
            # Stash data for deferred send
            self._pending_nccl_send = (data, keys_order)
            return encode_nccl_metadata(data, keys_order, cpu_scalars=cpu_scalars)
        # Default: compact wire format (no pickle)
        return _wire.encode_to_buffer(data)

    def _flush_nccl_send(self):
        """Send pending NCCL tensors (called AFTER HTTP response is flushed)."""
        pending = self._pending_nccl_send
        if pending is not None:
            self._pending_nccl_send = None
            data, keys_order = pending
            try:
                self._nccl_transport.send_tensors(data, keys_order)
            except Exception:
                logger.exception(
                    "NCCL send failed — disabling NCCL for subsequent requests"
                )
                transport = self._nccl_transport
                self._nccl_transport = None
                if transport is not None:
                    try:
                        transport.destroy()
                    except Exception:
                        logger.exception(
                            "Failed to destroy NCCL transport after send failure"
                        )
                raise

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    # -- Forward helpers (NCCL-safe, return raw dicts) --

    def _run_generate_eagle3_data(
        self, raw_body: bytes, rank_only_forward: bool = False
    ) -> dict:
        """Execute the full Eagle3 forward pass.

        If NCCL mode is enabled, tensors stay on GPU (sent via NCCL).
        Otherwise they are moved to CPU for serialization.

        Parameters
        ----------
        rank_only_forward : bool
            If True, only participate in the model forward (for TP allreduce)
            and skip post-processing.  Used by non-rank-0 workers whose
            results are discarded.
        """
        _t = {}

        _t0 = time.perf_counter()
        payload = _deserialize_tensors(raw_body, map_location="cpu")
        input_ids = payload["input_ids"].cuda()
        attention_mask = payload["attention_mask"].cuda()
        loss_mask = payload["loss_mask"].cuda()
        is_vlm = payload.get("is_vlm", torch.tensor(False)).item()
        shard_returns = bool(payload.get("shard_returns", torch.tensor(False)).item())
        if is_vlm and shard_returns:
            raise ValueError(
                "shard_returns is not supported for VLM Eagle3 target output"
            )
        _t["deser"] = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        if is_vlm:
            result = self._model.generate_eagle3_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                pixel_values=payload.get("pixel_values", None),
                image_grid_thw=payload.get("image_grid_thw", None),
                is_vlm=True,
                shard_returns=shard_returns,
                rank_only_forward=rank_only_forward and not shard_returns,
                pad_target=(self._vocab_t2d is None),
            )
        else:
            result = self._model.generate_eagle3_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                shard_returns=shard_returns,
                rank_only_forward=rank_only_forward and not shard_returns,
                pad_target=(self._vocab_t2d is None),
            )
        _t["model_fwd"] = time.perf_counter() - _t0

        # Non-rank-0 workers only need to participate in the model forward
        # (for TP allreduce).  Skip target_p computation and .cpu() transfers
        # since their result is discarded by the caller.  With shard_returns,
        # each server TP rank owns a different batch shard, so workers must
        # still post-process their shard before rank 0 gathers the response.
        if rank_only_forward and not shard_returns:
            parts = ", ".join(f"{k}={v:.3f}s" for k, v in _t.items())
            logger.debug("EAGLE3_WORKER_TIMING: %s", parts)
            return {}

        _t0 = time.perf_counter()
        target_tensor = result.target
        # Determine if tensors stay on GPU (NCCL path) or go to CPU
        keep_gpu = self._keep_on_gpu
        _maybe_cpu = (
            (lambda t: t)
            if keep_gpu
            else (lambda t: t.cpu() if t is not None else None)
        )

        if target_tensor is not None and self._vocab_t2d is not None:
            if (
                self._vocab_t2d_cuda is None
                or self._vocab_t2d_cuda.device != target_tensor.device
            ):
                self._vocab_t2d_cuda = self._vocab_t2d.to(target_tensor.device)
            t2d = self._vocab_t2d_cuda
            result_loss_mask = result.loss_mask
            _lm = (
                result_loss_mask.unsqueeze(-1)
                if result_loss_mask.dim() == 2
                else result_loss_mask
            )
            target_max_token = padding(target_tensor.argmax(-1), left=False)
            position_mask = t2d[target_max_token][..., None].int() * _lm
            target_head = target_tensor[..., t2d].float()
            target_p = torch.softmax(target_head, dim=-1).detach()
            target_p = torch.cat(
                (
                    target_p[:, 1:],
                    torch.full_like(target_p[:, -1:], 1 / target_p.shape[-1]),
                ),
                dim=1,
            )
            input_ids_out = padding(result.input_ids, left=False)

            # Top-k sparsification for transfer efficiency.
            seq_dim = target_p.shape[-1]  # draft vocab size
            if self._topk > 0 and self._topk < seq_dim:
                topk_vals, topk_indices = target_p.topk(self._topk, dim=-1)
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(
                    1e-8
                )
                target_out_vals = (
                    topk_vals.bfloat16() if self._use_bf16_target else topk_vals
                )
                # int16 is safe when draft_vocab_size <= 32768; use int32 otherwise
                idx_dtype = torch.int16 if seq_dim <= 32768 else torch.int32
                target_out_indices = topk_indices.to(idx_dtype)
                target_key = None
            else:
                target_out_vals = (
                    target_p.bfloat16() if self._use_bf16_target else target_p
                )
                target_out_indices = None
                target_key = target_out_vals

            out = {
                "hidden_states": _maybe_cpu(result.hidden_states),
                "loss_mask": _maybe_cpu(result.loss_mask),
                "input_ids": _maybe_cpu(input_ids_out),
                "position_mask": _maybe_cpu(position_mask),
            }
            if target_out_indices is not None:
                out["target_topk_vals"] = _maybe_cpu(target_out_vals)
                out["target_topk_indices"] = _maybe_cpu(target_out_indices)
                out["target_vocab_size"] = torch.tensor([seq_dim], dtype=torch.int32)
            else:
                out["target"] = _maybe_cpu(target_key)
        else:
            out = {
                "hidden_states": _maybe_cpu(result.hidden_states),
                "target": _maybe_cpu(target_tensor),
                "loss_mask": _maybe_cpu(result.loss_mask),
                "input_ids": _maybe_cpu(result.input_ids),
            }
        if result.last_hidden_states is not None:
            out["last_hidden_states"] = _maybe_cpu(result.last_hidden_states)
        _t["target_p"] = time.perf_counter() - _t0

        parts = ", ".join(f"{k}={v:.3f}s" for k, v in _t.items())
        logger.debug("EAGLE3_SERVER_TIMING: %s | TOTAL=%.3fs", parts, sum(_t.values()))
        return out

    def handle_generate_eagle3_data(self, raw_body: bytes) -> bytes:
        return self._serialize_response(self._run_generate_eagle3_data(raw_body))

    def _run_generate_dflash_data(
        self, raw_body: bytes, rank_only_forward: bool = False
    ) -> dict:
        """Execute the full DFlash forward pass.

        Parameters
        ----------
        rank_only_forward : bool
            If True, only participate in the model forward (for TP allreduce)
            and skip post-processing.  Used by non-rank-0 workers whose
            results are discarded.
        """
        payload = _deserialize_tensors(raw_body, map_location="cpu")
        input_ids = payload["input_ids"].cuda()
        attention_mask = payload["attention_mask"].cuda()
        loss_mask = payload["loss_mask"].cuda()

        result = self._model.generate_dflash_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )

        # Non-rank-0 workers only participate in model forward for TP allreduce.
        if rank_only_forward:
            return {}

        keep_gpu = self._keep_on_gpu
        _maybe_cpu = (
            (lambda t: t)
            if keep_gpu
            else (lambda t: t.cpu() if t is not None else None)
        )
        out = {
            "hidden_states": _maybe_cpu(result.hidden_states),
            "input_ids": _maybe_cpu(result.input_ids),
            "attention_mask": _maybe_cpu(result.attention_mask),
            "loss_mask": _maybe_cpu(result.loss_mask),
        }
        if result.position_ids is not None:
            out["position_ids"] = _maybe_cpu(result.position_ids)
        return out

    def handle_generate_dflash_data(self, raw_body: bytes) -> bytes:
        return self._serialize_response(self._run_generate_dflash_data(raw_body))

    def handle_set_aux_hidden_states_layers(self, raw_body: bytes) -> bytes:
        payload = _deserialize_tensors(raw_body, map_location="cpu")
        layers = payload["layers"].tolist()
        self._model.set_aux_hidden_states_layers(layers)
        logger.info("Eagle3 aux hidden state layers set to %s", layers)
        return b"ok"

    def handle_set_capture_layers(self, raw_body: bytes) -> bytes:
        payload = _deserialize_tensors(raw_body, map_location="cpu")
        layers = payload["layers"].tolist()
        self._model.set_capture_layers(layers)
        logger.info("DFlash capture layers set to %s", layers)
        return b"ok"

    def handle_set_vocab_mapping(self, raw_body: bytes) -> bytes:
        """Receive t2d/d2t vocab mapping for server-side target_p computation."""
        payload = _deserialize_tensors(raw_body, map_location="cpu")
        self._vocab_t2d = payload["t2d"]
        self._vocab_t2d_cuda = None  # invalidate cached CUDA version
        logger.info("Vocab mapping set: t2d shape=%s", self._vocab_t2d.shape)
        return b"ok"

    def handle_get_model_info(self, _raw_body: bytes) -> bytes:
        info = {}
        if self._model is not None:
            if hasattr(self._model, "model_runner"):
                runner = self._model.model_runner
                if hasattr(runner, "model_config") and hasattr(
                    runner.model_config, "hf_config"
                ):
                    hf_cfg = runner.model_config.hf_config
                    if hf_cfg is not None and hasattr(hf_cfg, "num_hidden_layers"):
                        info["num_hidden_layers"] = hf_cfg.num_hidden_layers
                    if hf_cfg is not None and hasattr(hf_cfg, "to_dict"):
                        info["hf_config_dict"] = hf_cfg.to_dict()
        info["server_model_path"] = self.model_path
        info["mode"] = self.mode
        return json.dumps(info, default=str).encode()

    def handle_init_nccl(self, raw_body: bytes) -> bytes:
        """Initialize NCCL transport on the server side.

        Called by the client via POST /init_nccl.  The client provides
        the NCCL port in the request body (JSON).  Both sides block on
        the TCP store rendezvous until the group is established.
        """
        if not self._nccl_enabled:
            return json.dumps({"status": "disabled"}).encode()

        try:
            request_data = json.loads(raw_body.decode("utf-8"))
            nccl_port = request_data.get("nccl_port", self._nccl_port)
        except (json.JSONDecodeError, UnicodeDecodeError):
            nccl_port = self._nccl_port

        if self._nccl_transport is not None and self._nccl_transport.is_initialized:
            return json.dumps({"status": "already_initialized"}).encode()

        # Resolve NCCL port
        if nccl_port is None:
            return json.dumps(
                {"status": "error", "message": "No NCCL port specified"}
            ).encode()

        # Create and initialize NCCL transport (server = rank 0).  When the
        # HTTP server binds to a wildcard address, make TCPStore listen on all
        # interfaces; clients connect via the URL host/IP they were given.
        nccl_host = "0.0.0.0" if self._host in ("0.0.0.0", "::") else self._host
        self._nccl_transport = NCCLTransport(
            nccl_port=nccl_port,
            host=nccl_host,
            is_server=True,
        )

        success = self._nccl_transport.initialize(timeout_seconds=120)
        if success:
            return json.dumps({"status": "ok"}).encode()
        else:
            self._nccl_transport = None
            return json.dumps(
                {"status": "error", "message": "NCCL init failed"}
            ).encode()


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threading HTTP server with concurrency cap to prevent OOM from
    multiple concurrent forward passes."""

    daemon_threads = True


# Global semaphore to limit concurrent GPU-intensive request handlers.
# Set to 1 because: (a) NCCL broadcast must not be interleaved across threads
# when tp > 1, and (b) single-GPU forward + serialisation already saturates
# one CPU core – real parallelism comes from multi-server via --remote-urls.
_forward_semaphore = threading.BoundedSemaphore(1)
_request_local = threading.local()


class _RequestHandler(BaseHTTPRequestHandler):
    """Minimal handler that dispatches to TargetModelServer methods."""

    # Reference set by the factory before starting the server
    server_app: TargetModelServer = None

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _send_response(self, body: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        # Echo the transport mode so the client knows how to decode
        if self.server_app._keep_on_gpu:
            self.send_header(NCCL_HEADER, "1")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        # Deferred NCCL send: after the HTTP response is flushed, send
        # tensors via NCCL so the client can recv after decoding metadata.
        self.server_app._flush_nccl_send()

    def _send_error(self, message: str, status: int = 500):
        data = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_response(b"ok")
        else:
            self._send_error("Not found", 404)

    def do_POST(self):
        try:
            try:
                body = self._read_body()
            except Exception:
                self._send_error("Failed to read request body", 400)
                return

            path = self.path.rstrip("/")

            if path in _CLIENT_ACTIVITY_ENDPOINTS:
                self.server_app._record_client_activity()

            # Worker threads spawned by ThreadingHTTPServer do not inherit the
            # main thread's CUDA device.  Set it here so all GPU ops (Triton
            # kernels, tensor allocations, etc.) target the correct device.
            if path not in _LIFECYCLE_ENDPOINTS and self.server_app._gpu_id is not None:
                torch.cuda.set_device(self.server_app._gpu_id)

            # Detect transport mode from headers (request-local)
            use_nccl = self.headers.get(NCCL_HEADER) == "1"
            _request_local.use_nccl = use_nccl

            # Heavy endpoints: gate concurrent forward passes AND response/NCCL send.
            if path in ("/generate_eagle3_data", "/generate_dflash_data"):
                acquired = _forward_semaphore.acquire(blocking=False)
                if not acquired:
                    self._send_error(
                        "Server busy – too many concurrent forward passes", 503
                    )
                    return
                try:
                    result = _route_request_synced(self.server_app, path, body)
                    if result is not None:
                        if isinstance(result, tuple):
                            resp_body, status = result
                            self._send_response(resp_body, status)
                        else:
                            self._send_response(result)
                finally:
                    self.server_app._pending_nccl_send = None
                    _forward_semaphore.release()
            else:
                result = _route_request_synced(self.server_app, path, body)
                if result is not None:
                    if isinstance(result, tuple):
                        resp_body, status = result
                        self._send_response(resp_body, status)
                    else:
                        self._send_response(result)
        finally:
            _request_local.use_nccl = False

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


def make_handler(app: TargetModelServer):
    """Create a handler class bound to a specific TargetModelServer."""

    class Handler(_RequestHandler):
        server_app = app

    return Handler


def create_http_server(
    server_app: TargetModelServer, host: str, port: int
) -> HTTPServer:
    """Create and return a threading HTTP server instance."""
    handler = make_handler(server_app)
    httpd = _ThreadingHTTPServer((host, port), handler)
    return httpd


# ---------------------------------------------------------------------------
# Synchronous request routing (for tp > 1)
# ---------------------------------------------------------------------------

_SENTINEL_EXIT = "__EXIT__"


def _route_request(app, path, body):
    """Dispatch a POST request to the appropriate handler. Returns bytes or raises."""
    if path == "/generate_eagle3_data":
        return app.handle_generate_eagle3_data(body)
    elif path == "/generate_dflash_data":
        return app.handle_generate_dflash_data(body)
    elif path == "/init_nccl":
        return app.handle_init_nccl(body)
    elif path == "/set_aux_hidden_states_layers":
        return app.handle_set_aux_hidden_states_layers(body)
    elif path == "/set_capture_layers":
        return app.handle_set_capture_layers(body)
    elif path == "/get_model_info":
        return app.handle_get_model_info(body)
    elif path == "/set_vocab_mapping":
        return app.handle_set_vocab_mapping(body)
    elif path == "/disconnect":
        return app.handle_disconnect(body)
    elif path == "/heartbeat":
        return app.handle_heartbeat(body)
    else:
        raise ValueError(f"Unknown endpoint: {path}")


_EAGLE3_BATCH_OUTPUT_KEYS = {
    "hidden_states",
    "target",
    "target_topk_vals",
    "target_topk_indices",
    "loss_mask",
    "input_ids",
    "position_mask",
    "last_hidden_states",
}


def _all_ranks_succeeded(success: bool) -> bool:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return success
    flag = torch.tensor([1 if success else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _gather_tensor_dict_to_rank0(data: dict | None) -> dict | None:
    """Gather per-rank tensor dict shards along batch dim to rank 0."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return data
    if not _all_ranks_succeeded(data is not None):
        raise ValueError(
            "Cannot gather Eagle3 sharded output: at least one rank has no data"
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    keys = None
    if rank == 0:
        keys = [
            k
            for k, v in data.items()
            if k in _EAGLE3_BATCH_OUTPUT_KEYS and isinstance(v, torch.Tensor)
        ]
    keys_list = [keys]
    dist.broadcast_object_list(keys_list, src=0)
    keys = keys_list[0]

    gathered = {} if rank == 0 else None
    for key in keys:
        local = data[key]
        shape = torch.tensor(local.shape, dtype=torch.long, device="cuda")
        ndim = torch.tensor([local.ndim], dtype=torch.long, device="cuda")
        dist.broadcast(ndim, src=0)
        if rank != 0:
            shape = torch.empty(int(ndim.item()), dtype=torch.long, device="cuda")
        dist.broadcast(shape, src=0)
        shape_tuple = tuple(int(x) for x in shape.cpu().tolist())
        if tuple(local.shape) != shape_tuple:
            raise ValueError(
                f"Cannot gather remote sharded output {key}: rank {rank} shape "
                f"{tuple(local.shape)} != rank0 shape {shape_tuple}"
            )

        local_device = local.device
        local_for_gather = local.contiguous()
        if not local_for_gather.is_cuda:
            local_for_gather = local_for_gather.cuda()
        chunks = (
            [torch.empty_like(local_for_gather) for _ in range(world_size)]
            if rank == 0
            else None
        )
        dist.gather(local_for_gather, gather_list=chunks, dst=0)
        if rank == 0:
            gathered_value = torch.cat(chunks, dim=0)
            if local_device.type == "cpu":
                gathered_value = gathered_value.cpu()
            gathered[key] = gathered_value

    if rank == 0:
        for key, value in data.items():
            if key not in keys:
                gathered[key] = value
    return gathered


def _broadcast_request(path, body):
    """Broadcast a POST request (path + body) from rank 0 to all tp ranks.

    Returns (path, body_bytes) on all ranks.  Tensors are placed on CUDA so
    they are compatible with the NCCL backend.
    """
    rank = dist.get_rank()

    # Broadcast endpoint path
    path_list = [path]
    dist.broadcast_object_list(path_list, src=0)

    # Broadcast body length + data (CUDA tensors required for NCCL)
    if rank == 0:
        body_len = len(body)
        body_len_t = torch.tensor([body_len], dtype=torch.long, device="cuda")
        if body_len > 0:
            body_tensor = torch.frombuffer(bytearray(body), dtype=torch.uint8).cuda()
        else:
            body_tensor = torch.empty(0, dtype=torch.uint8, device="cuda")
    else:
        body_len_t = torch.zeros(1, dtype=torch.long, device="cuda")

    dist.broadcast(body_len_t, src=0)

    if rank != 0:
        blen = body_len_t.item()
        body_tensor = (
            torch.zeros(blen, dtype=torch.uint8, device="cuda")
            if blen > 0
            else torch.empty(0, dtype=torch.uint8, device="cuda")
        )
    dist.broadcast(body_tensor, src=0)

    body_bytes = body_tensor.cpu().numpy().tobytes() if body_tensor.numel() > 0 else b""
    return path_list[0], body_bytes


def _route_request_synced(app, path, body):
    """Route a request, broadcasting across tp ranks so all ranks participate in the
    forward pass (required by NCCL allreduce inside SGLang model_runner).

    For heavy forward-pass endpoints a rank-0-only serialisation path is used
    so that worker ranks do not waste CPU serialising results that are discarded.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        try:
            return _route_request(app, path, body)
        except Exception:
            logger.exception("Error handling %s", path)
            return json.dumps({"error": "Internal server error"}).encode(), 500

    # /init_nccl must run ONLY on rank 0 — it sets up a separate 2-rank NCCL
    # group (server rank 0 + training client rank 1).  TP worker ranks must
    # NOT participate.  Same for /disconnect and /heartbeat.
    if path in ("/init_nccl", "/disconnect", "/heartbeat"):
        return _route_request(app, path, body)

    rank = dist.get_rank()
    synced_path, body_bytes = _broadcast_request(path, body)

    # Heavy endpoints: all ranks execute the forward pass (NCCL requires it),
    # but only rank 0 needs post-processing (target_p, serialisation).
    # Non-rank-0 workers skip post-processing to avoid blocking the next step.
    if synced_path == "/generate_eagle3_data":
        out = None
        shard_returns = False
        forward_ok = False
        try:
            payload = _deserialize_tensors(body_bytes, map_location="cpu")
            shard_returns = bool(
                payload.get("shard_returns", torch.tensor(False)).item()
            )
            out = app._run_generate_eagle3_data(
                body_bytes, rank_only_forward=(rank != 0)
            )
            forward_ok = out is not None
        except Exception:
            logger.exception("Error handling %s (rank %d)", synced_path, rank)
        if shard_returns:
            try:
                if _all_ranks_succeeded(forward_ok):
                    out = _gather_tensor_dict_to_rank0(out)
                else:
                    out = None
            except Exception:
                logger.exception(
                    "Error gathering %s shards (rank %d)", synced_path, rank
                )
                out = None
        if rank == 0:
            if out is not None:
                return app._serialize_response(out)
            return (
                json.dumps(
                    {"error": "Internal server error during forward pass"}
                ).encode(),
                500,
            )
        return None

    if synced_path == "/generate_dflash_data":
        out = None
        try:
            out = app._run_generate_dflash_data(
                body_bytes, rank_only_forward=(rank != 0)
            )
        except Exception:
            logger.exception("Error handling %s (rank %d)", synced_path, rank)
        if rank == 0:
            if out is not None:
                return app._serialize_response(out)
            return (
                json.dumps(
                    {"error": "Internal server error during forward pass"}
                ).encode(),
                500,
            )
        return None

    # Lightweight endpoints: broadcast + execute on all ranks
    try:
        result = _route_request(app, synced_path, body_bytes)
    except Exception:
        logger.exception("Error handling %s (rank %d)", synced_path, rank)
        result = None

    if rank == 0:
        if result is not None:
            return result
        return json.dumps({"error": "Internal server error"}).encode(), 500
    return None


def _worker_loop(server_app):
    """Non-rank-0 worker loop for tp > 1.

    Waits for requests broadcast by rank 0 and participates in every forward pass
    via the shared NCCL process group.  Results are computed but discarded — only
    rank 0 sends the HTTP response.
    """
    rank = dist.get_rank()
    logger.info("Worker rank %d entering sync loop", rank)

    while True:
        # Step 1: receive path (matches rank 0's first broadcast inside _broadcast_request)
        path_list = [""]
        dist.broadcast_object_list(path_list, src=0)
        path = path_list[0]

        if path == _SENTINEL_EXIT:
            logger.info("Worker rank %d received exit signal", rank)
            break

        # Step 2: receive body_len + body (matches rank 0's remaining broadcasts
        # inside _broadcast_request — do NOT call _broadcast_request again as it
        # would re-broadcast path, causing misalignment)
        body_len_t = torch.zeros(1, dtype=torch.long, device="cuda")
        dist.broadcast(body_len_t, src=0)
        blen = body_len_t.item()
        body_tensor = (
            torch.zeros(blen, dtype=torch.uint8, device="cuda")
            if blen > 0
            else torch.empty(0, dtype=torch.uint8, device="cuda")
        )
        dist.broadcast(body_tensor, src=0)
        body_bytes = (
            body_tensor.cpu().numpy().tobytes() if body_tensor.numel() > 0 else b""
        )

        # Heavy endpoints: worker only participates in model forward (TP allreduce),
        # skips post-processing (target_p, .cpu() transfers) since result is discarded.
        if path == "/generate_eagle3_data":
            out = None
            shard_returns = False
            forward_ok = False
            try:
                payload = _deserialize_tensors(body_bytes, map_location="cpu")
                shard_returns = bool(
                    payload.get("shard_returns", torch.tensor(False)).item()
                )
                out = server_app._run_generate_eagle3_data(
                    body_bytes, rank_only_forward=True
                )
                forward_ok = out is not None
            except Exception:
                logger.exception("Worker rank %d error handling %s", rank, path)
            if shard_returns:
                try:
                    if _all_ranks_succeeded(forward_ok):
                        _gather_tensor_dict_to_rank0(out)
                except Exception:
                    logger.exception(
                        "Worker rank %d error gathering %s shards", rank, path
                    )
        else:
            try:
                if path == "/generate_dflash_data":
                    server_app._run_generate_dflash_data(
                        body_bytes, rank_only_forward=True
                    )
                else:
                    _route_request(server_app, path, body_bytes)
            except Exception:
                logger.exception("Worker rank %d error handling %s", rank, path)
