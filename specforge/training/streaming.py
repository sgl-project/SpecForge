"""Persistent streaming chunk source.

Invariant: a chunk is a *data-delivery unit* only —
  chunk boundary != process boundary
  chunk boundary != optimizer boundary
  chunk boundary != mandatory checkpoint boundary
One chunk is consumed as one *epoch* of the persistent trainer: the controller
already calls ``data.set_epoch(epoch)`` at every epoch top, which routes here.
Model, FSDP state, optimizer (fp32 masters + Adam moments), scheduler,
global_step, RNG, CUDA context, NCCL group and compiled kernels all live for
the whole run.
"""
import glob
import logging
import os
import shutil
import time

import torch
import torch.distributed as dist

def _say(msg):
    print(f"[streaming] {msg}", flush=True)
    logger.info(msg)

logger = logging.getLogger("specforge.streaming")


class StreamingChunkSource:
    """Blocks until the next READY chunk; pads refs to a fixed per-epoch size.

    Ready protocol (producer side, already deployed): rows are written as
    ``t*.pt.tmp`` + atomic rename, then ``READY.marker`` is written last.
    A chunk directory without READY.marker is invisible here (race-safe).
    Consumption protocol: CONSUMED marker then directory removal, rank0 only,
    after a barrier proves every rank finished the epoch that used it.
    """

    def __init__(
        self,
        *,
        provider,
        stream_dir: str,
        run_id: str,
        ttt_length: int,
        max_len: int,
        chunk_rows: int,
        poll_s: float = 0.5,
    ):
        self.provider = provider
        self.dir = stream_dir
        self.run_id = run_id
        self.ttt = ttt_length
        self.max_len = max_len
        self.rows = int(chunk_rows)
        self.poll = poll_s
        self.cur = None
        self.chunks_done = 0
        self._export = None
        self._memo_epoch = None
        self._memo_refs = None
        _say(f"streaming source ready: stream_dir={stream_dir} chunk_rows={self.rows} (persistent trainer)")

    # -- optional lightweight export at chunk boundaries ---------------------
    def bind_export(self, *, model, export_dir: str, every_chunks: int, get_step=None):
        self._export = (model, export_dir, max(0, int(every_chunks)), get_step)

    # -- internals ------------------------------------------------------------
    def _rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    def _bcast(self, obj):
        if not dist.is_initialized():
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _next_ready_rank0(self):
        t0 = time.perf_counter()
        while True:
            for c in sorted(glob.glob(os.path.join(self.dir, "chunk_*"))):
                if os.path.exists(os.path.join(c, "READY.marker")) and not os.path.exists(
                    os.path.join(c, "CONSUMED")
                ):
                    return c, time.perf_counter() - t0
            time.sleep(self.poll)

    def _consume_previous(self):
        if self.cur is None:
            return
        if self._rank() == 0:
            try:
                open(os.path.join(self.cur, "CONSUMED"), "w").close()
                shutil.rmtree(self.cur, ignore_errors=True)
            except OSError:
                pass
        self.cur = None

    def _maybe_export(self):
        if not self._export:
            return
        model, out, every, get_step = self._export
        if every <= 0 or self.chunks_done == 0 or self.chunks_done % every:
            return
        t0 = time.perf_counter()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        draft = getattr(model, "draft_model", model)
        root = None
        if isinstance(model, FSDP):
            root = model
        elif isinstance(draft, FSDP):
            root = draft
        else:
            for m in getattr(model, "modules", lambda: [])():
                if isinstance(m, FSDP):
                    root = m
                    break
        if root is not None:
            cfgs = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(root, StateDictType.FULL_STATE_DICT, cfgs):
                sd = draft.state_dict()
        else:
            sd = draft.state_dict()
        if self._rank() == 0:
            from safetensors.torch import save_file

            sd = {
                k: v.detach().to(torch.bfloat16).contiguous()
                for k, v in sd.items()
                if isinstance(v, torch.Tensor)
            }
            os.makedirs(out, exist_ok=True)
            step = -1
            if get_step is not None:
                try:
                    step = int(get_step())
                except Exception:
                    pass
            tmp = os.path.join(out, "model.safetensors.tmp")
            save_file(sd, tmp)
            os.replace(tmp, os.path.join(out, "model.safetensors"))
            _say(f"export_done chunks={self.chunks_done} global_step={step} export_time={time.perf_counter()-t0:.1f}s dir={out}")
        if dist.is_initialized():
            dist.barrier()

    def peek_refs(self):
        """Build-time sizing ONLY: read the first READY chunk without consuming it.
        The controller calls set_epoch(start_epoch) before any batch is read, so
        these refs are placeholders; the real epoch grab happens there (and on
        resume, start_epoch reflects the restored position)."""
        nxt = None
        wait = 0.0
        if self._rank() == 0:
            nxt, wait = self._next_ready_rank0()
        nxt = self._bcast(nxt)
        refs = list(
            self.provider.build_reader(
                nxt, run_id=self.run_id, ttt_length=self.ttt, max_len=self.max_len
            ).read()
        )
        n = len(refs)
        if n < self.rows:
            refs = (refs * ((self.rows + n - 1) // n))[: self.rows]
        elif n > self.rows:
            refs = refs[: self.rows]
        if self._rank() == 0:
            _say(f"peek(build sizing): chunk={os.path.basename(nxt)} rows={n} wait={wait:.1f}s (not consumed)")
        return refs

    # -- API: one call per epoch from the persistent fit loop ------------------
    def next_chunk_refs(self, epoch: int):
        if epoch == self._memo_epoch:
            return self._memo_refs  # launch-time priming + set_epoch(0) dedupe
        t_bound = time.perf_counter()
        if dist.is_initialized():
            dist.barrier()  # every rank finished the previous chunk
        if self.cur is not None:
            self.chunks_done += 1
        self._maybe_export()
        self._consume_previous()
        wait = 0.0
        nxt = None
        if self._rank() == 0:
            nxt, wait = self._next_ready_rank0()
        nxt = self._bcast(nxt)
        self.cur = nxt
        refs = list(
            self.provider.build_reader(
                nxt, run_id=self.run_id, ttt_length=self.ttt, max_len=self.max_len
            ).read()
        )
        if not refs:
            raise ValueError(f"stream chunk {nxt} produced no refs")
        n = len(refs)
        if n < self.rows:  # cycle-pad so every epoch has the fixed ref count
            refs = (refs * ((self.rows + n - 1) // n))[: self.rows]
        elif n > self.rows:
            refs = refs[: self.rows]
        if self._rank() == 0:
            _say(f"chunk_start epoch={epoch} chunk={os.path.basename(nxt)} rows={n} padded_to={self.rows} wait_for_data_time={wait:.1f}s boundary_overhead={time.perf_counter()-t_bound-wait:.1f}s chunks_done={self.chunks_done}")
        self._memo_epoch, self._memo_refs = epoch, refs
        return refs
