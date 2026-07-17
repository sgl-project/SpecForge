import logging

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)

_ADAMW_BACKENDS = frozenset({"torch", "fused"})


class BF16Optimizer:
    """AdamW over fp32 master copies of the bf16 trainable params, with grad
    clipping and cosine warmup scheduling."""

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        adamw_backend="torch",
    ):
        # defaults copied from EAGLE traineagle3 ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        if adamw_backend not in _ADAMW_BACKENDS:
            raise ValueError(
                f"adamw_backend={adamw_backend!r}; must be one of "
                f"{sorted(_ADAMW_BACKENDS)}"
            )
        if adamw_backend == "fused":
            devices = {param.device for param in self.fp32_params}
            if (
                len(devices) != 1
                or next(iter(devices), torch.device("cpu")).type != "cuda"
            ):
                raise ValueError(
                    "adamw_backend='fused' requires trainable parameters on one "
                    "CUDA device"
                )
            self.optimizer = torch.optim.AdamW(
                self.fp32_params,
                lr=lr,
                weight_decay=weight_decay,
                foreach=False,
                fused=True,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.fp32_params, lr=lr, weight_decay=weight_decay
            )
        self.adamw_backend = adamw_backend
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled

    def _global_grad_norm(self, grads, *, foreach: bool = False):
        """Return the L2 norm across every rank-local gradient shard."""
        if grads:
            if foreach:
                local_norm = torch.nn.utils.get_total_norm(grads, foreach=True)
                total_norm_sq = local_norm.square()
            else:
                total_norm_sq = torch.stack(
                    [grad.square().sum() for grad in grads]
                ).sum()
        else:
            device = self.fp32_params[0].device if self.fp32_params else "cpu"
            total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)

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

        return total_norm_sq.sqrt()

    def _clip_grad_norm(self):
        """Clip all FSDP shards with one global L2-norm coefficient."""
        grads = [mp.grad for mp in self.fp32_params if mp.grad is not None]
        total_norm = self._global_grad_norm(grads)

        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        for grad in grads:
            grad.mul_(clip_coef)
        return total_norm

    def _fused_adamw_step(self):
        gradients = [mp.grad for mp in self.fp32_params if mp.grad is not None]
        grad_norm = self._global_grad_norm(gradients, foreach=True)
        if gradients:
            self.optimizer.grad_scale = ((grad_norm + 1e-6) / self.max_grad_norm).clamp(
                min=1.0
            )
        try:
            self.optimizer.step()
        finally:
            if hasattr(self.optimizer, "grad_scale"):
                del self.optimizer.grad_scale
        return grad_norm

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        if self.adamw_backend == "fused":
            grad_norm = self._fused_adamw_step()
        else:
            grad_norm = self._clip_grad_norm()
            self.optimizer.step()
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return self.last_grad_norm

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        saved_max_grad_norm = state_dict.get("max_grad_norm")
        if saved_max_grad_norm is not None and float(saved_max_grad_norm) != float(
            self.max_grad_norm
        ):
            raise ValueError(
                "checkpoint optimizer used max_grad_norm="
                f"{saved_max_grad_norm} but this run has "
                f"max_grad_norm={self.max_grad_norm}"
            )
        saved_backend = state_dict.get("adamw_backend", "torch")
        if saved_backend not in _ADAMW_BACKENDS:
            raise ValueError(f"checkpoint has invalid adamw_backend={saved_backend!r}")
        if saved_backend != self.adamw_backend:
            raise ValueError(
                "checkpoint adamw_backend does not match this optimizer "
                f"(checkpoint={saved_backend!r}, current={self.adamw_backend!r})"
            )
        optimizer_state = state_dict["optimizer_state_dict"]
        param_groups = optimizer_state.get("param_groups", [])
        if saved_backend == "fused" and any(
            group.get("foreach") is not False or group.get("fused") is not True
            for group in param_groups
        ):
            raise ValueError(
                "fused AdamW checkpoint does not carry the fused execution flags"
            )
        if saved_backend == "torch" and any(
            group.get("fused") is True for group in param_groups
        ):
            raise ValueError(
                "torch AdamW checkpoint unexpectedly enables the fused backend"
            )
        self.optimizer.load_state_dict(optimizer_state)
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
                    mp.data.copy_(p.detach().to(torch.float32))

    def state_dict(self):
        return {
            "adamw_backend": self.adamw_backend,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "max_grad_norm": self.max_grad_norm,
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
