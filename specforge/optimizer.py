import logging

import torch
import torch.distributed as dist

from specforge.lr_scheduler import ConstantWarmupLR, CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


def _pinned_like(model_param, dtype):
    """A host tensor for staging transfers of ``model_param``; pinned when the
    model lives on an accelerator (non-blocking copies need page-locked memory)."""
    pin = model_param.device.type == "cuda" and torch.cuda.is_available()
    return torch.empty(model_param.shape, dtype=dtype, pin_memory=pin)


class BF16Optimizer:
    """AdamW over fp32 master copies of the bf16 trainable params, with grad
    clipping and configurable warmup scheduling.

    With ``offload_master`` the masters and Adam moments live on the host. The
    step then runs the host round trip through persistent page-locked
    buffers: gradients are cast to fp32 and clipped on the device, copied
    asynchronously into pinned fp32 master grads, updated by torch's fused
    CPU AdamW, converted to the model dtype into a pinned staging buffer and
    copied back. This matters at scale: for a ~5B-parameter shard per rank
    the naive path (pageable bf16 copy, host cast, single-tensor Adam, fresh
    20 GB allocations every step) costs ~14 s per optimizer step, more than
    the forward/backward of the whole accumulation window.
    """

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        lr_scheduler="cosine",
        offload_master=False,
    ):
        # defaults copied from EAGLE traineagle3 ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.offload_master = bool(offload_master)
        self.fp32_params = [
            (
                p.detach().to(device="cpu", dtype=torch.float32).clone()
                if self.offload_master
                else p.detach().clone().to(torch.float32)
            )
            for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        # Persistent host buffers for the offload round trip (see class doc).
        self._master_grad_buffers = (
            [_pinned_like(p, torch.float32) for p in self.model_params]
            if self.offload_master
            else None
        )
        self._staging_buffers = (
            [_pinned_like(p, p.dtype) for p in self.model_params]
            if self.offload_master
            else None
        )
        self.optimizer = self._build_adamw(lr, weight_decay)
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        scheduler_types = {
            "constant": ConstantWarmupLR,
            "cosine": CosineAnnealingWarmupLR,
        }
        if lr_scheduler not in scheduler_types:
            raise ValueError(
                f"unsupported lr_scheduler={lr_scheduler!r}; "
                f"expected one of {sorted(scheduler_types)}"
            )
        self.lr_scheduler_type = lr_scheduler
        self.scheduler = scheduler_types[lr_scheduler](
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def _build_adamw(self, lr, weight_decay):
        if self.offload_master and self.fp32_params:
            # torch's fused CPU AdamW is one multi-threaded pass over
            # param/grad/moments; the single-tensor path is ~15x slower.
            try:
                return torch.optim.AdamW(
                    self.fp32_params, lr=lr, weight_decay=weight_decay, fused=True
                )
            except (RuntimeError, ValueError) as exc:  # pragma: no cover
                logger.warning("fused CPU AdamW unavailable (%s); using foreach", exc)
                return torch.optim.AdamW(
                    self.fp32_params, lr=lr, weight_decay=weight_decay, foreach=True
                )
        return torch.optim.AdamW(self.fp32_params, lr=lr, weight_decay=weight_decay)

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled

    def _reduce_grad_norm(self, total_norm_sq):
        """All-reduce the squared L2 norm across shard ranks and derive the
        clip coefficient.

        ``total_norm_sq`` must already live on a device the process group can
        reduce (e.g. CUDA for NCCL). Returns ``(total_norm, clip_coef)``.
        """
        if (
            self._reduce_grad_norm_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(
                total_norm_sq,
                op=dist.ReduceOp.SUM,
                group=self._grad_norm_process_group,
            )
        total_norm = total_norm_sq.sqrt()
        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        return total_norm, clip_coef

    def _grad_norm_and_clip_coefficient(self):
        """Compute the global grad norm from the model params on their own
        device, where NCCL can reduce it safely, without materialising master
        gradients first."""
        grads = [p.grad.detach() for p in self.model_params if p.grad is not None]
        if grads:
            total_norm_sq = torch.stack(
                [grad.float().square().sum() for grad in grads]
            ).sum()
        else:
            device = self.model_params[0].device if self.model_params else "cpu"
            total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
        return self._reduce_grad_norm(total_norm_sq)

    def _clip_grad_norm(self):
        """Clip already-populated FP32 master gradients in place.

        Convenience entry point for optimizer tests and custom loops. When
        masters are CPU-offloaded, only the scalar norm is moved to the model
        device so a NCCL process group can still participate in the reduction.
        """
        grads = [master.grad for master in self.fp32_params if master.grad is not None]
        if grads:
            local_norm_sq = torch.stack(
                [grad.float().square().sum() for grad in grads]
            ).sum()
        else:
            master_device = self.fp32_params[0].device if self.fp32_params else "cpu"
            local_norm_sq = torch.zeros((), dtype=torch.float32, device=master_device)

        reduction_device = (
            self.model_params[0].device if self.model_params else local_norm_sq.device
        )
        total_norm, clip_coef = self._reduce_grad_norm(
            local_norm_sq.to(reduction_device)
        )
        for grad in grads:
            coefficient = (
                clip_coef
                if clip_coef.device == grad.device
                else float(clip_coef.item())
            )
            grad.mul_(coefficient)
        return total_norm

    def step(self):
        grad_norm, clip_coefficient = self._grad_norm_and_clip_coefficient()
        if not bool(torch.isfinite(grad_norm)):
            # Clipping cannot rescale a non-finite norm and one such Adam step
            # NaNs the weights permanently. The norm is already all-reduced,
            # so every rank skips this step deterministically.
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.warning(
                    "skipping optimizer step: non-finite global grad norm "
                    f"(max_grad_norm={self.max_grad_norm})"
                )
            with torch.no_grad():
                for p in self.model_params:
                    p.grad = None
                for mp in self.fp32_params:
                    mp.grad = None
            self.last_grad_norm = grad_norm.detach()
            return self.last_grad_norm
        self.last_grad_norm = grad_norm.detach()
        if self.offload_master:
            self._offload_step(clip_coefficient)
        else:
            self._resident_step(clip_coefficient)
        return self.last_grad_norm

    def _resident_step(self, clip_coefficient):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                if p.grad is None:
                    mp.grad = None
                    continue
                master_grad = p.grad.detach().to(device=mp.device, dtype=torch.float32)
                master_grad.mul_(clip_coefficient)
                mp.grad = master_grad
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
                p.grad = None

    def _offload_step(self, clip_coefficient):
        """Host round trip through the persistent pinned buffers."""
        with torch.no_grad():
            for p, mp, grad_buffer in zip(
                self.model_params, self.fp32_params, self._master_grad_buffers
            ):
                if p.grad is None:
                    mp.grad = None
                    continue
                # Cast and clip on the device (one fused pass on the grad; the
                # fp32 temporary is stream-ordered with the copy, so it can be
                # released immediately), then stream the fp32 grad to the host.
                clipped = p.grad.detach().to(torch.float32)
                clipped.mul_(clip_coefficient)
                grad_buffer.copy_(clipped, non_blocking=True)
                del clipped
                mp.grad = grad_buffer
                p.grad = None
            if self.model_params and self.model_params[0].device.type == "cuda":
                torch.cuda.current_stream(self.model_params[0].device).synchronize()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp, staging in zip(
                self.model_params, self.fp32_params, self._staging_buffers
            ):
                if staging.device == p.device:
                    p.data.copy_(mp.data)
                    continue
                # Host-side cast into pinned memory, then an async upload.
                staging.copy_(mp.data)
                p.data.copy_(staging, non_blocking=True)
            if self.model_params and self.model_params[0].device.type == "cuda":
                # The staging buffers are reused next step; make sure the
                # uploads have landed before anything else touches them.
                torch.cuda.current_stream(self.model_params[0].device).synchronize()

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        saved_scheduler_type = state_dict.get("lr_scheduler_type", "cosine")
        if saved_scheduler_type != self.lr_scheduler_type:
            raise ValueError(
                "checkpoint optimizer used lr_scheduler="
                f"{saved_scheduler_type!r} but this run has "
                f"lr_scheduler={self.lr_scheduler_type!r}"
            )
        saved_max_grad_norm = state_dict.get("max_grad_norm")
        if saved_max_grad_norm is not None and float(saved_max_grad_norm) != float(
            self.max_grad_norm
        ):
            raise ValueError(
                "checkpoint optimizer used max_grad_norm="
                f"{saved_max_grad_norm} but this run has "
                f"max_grad_norm={self.max_grad_norm}"
            )
        # offload_master is a pure device-placement choice: restored fp32
        # masters and Adam moments are relocated to the current master device,
        # so toggling it on resume is safe and intentionally not gated here.
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")
        saved_fp32 = state_dict.get("fp32_params")
        if saved_fp32 is not None:
            if len(saved_fp32) != len(self.fp32_params):
                raise ValueError(
                    f"checkpoint carries {len(saved_fp32)} fp32 master params "
                    f"but this rank has {len(self.fp32_params)}"
                )
            with torch.no_grad():
                for i, (saved, mp) in enumerate(zip(saved_fp32, self.fp32_params)):
                    if saved.shape != mp.shape:
                        raise ValueError(
                            f"fp32 master param {i} shape mismatch: checkpoint "
                            f"{tuple(saved.shape)} vs current {tuple(mp.shape)}"
                        )
                    mp.data.copy_(saved.to(mp.device, mp.dtype))
        else:
            logger.warning(
                "checkpoint has no fp32_params; re-cloning master params from "
                "bf16 weights — resume will not be numerically faithful"
            )
            with torch.no_grad():
                for p, mp in zip(self.model_params, self.fp32_params):
                    mp.data.copy_(p.detach().to(device=mp.device, dtype=mp.dtype))

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
