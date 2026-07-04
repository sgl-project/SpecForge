import logging

import torch

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


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
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.last_grad_norm = None
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return self.last_grad_norm

    def _optimizer_state_matches(self, opt_state_dict) -> bool:
        # Under FSDP(use_orig_params=True) the AdamW momentum lives on fp32 copies
        # of the *per-rank-sharded* params, but save_checkpoint() only persists
        # rank-0's shard. Loading that onto other ranks blows up at optimizer.step()
        # with a size mismatch. Only accept the saved momentum if every stored
        # buffer matches this rank's current param sizes (true for single-rank /
        # matching-shard resumes); otherwise we reset and let Adam re-warm.
        state = opt_state_dict.get("state") or {}
        if not state:
            return False
        try:
            for idx, st in state.items():
                i = int(idx)
                if i >= len(self.fp32_params):
                    return False
                ea = st.get("exp_avg")
                if ea is not None and ea.numel() != self.fp32_params[i].numel():
                    return False
            return True
        except Exception:
            return False

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        # Always restore the LR schedule (rank-agnostic).
        if "scheduler_state_dict" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            print_on_rank0("Successfully loaded scheduler state_dict.")
        # Under FSDP(use_orig_params=True) only rank-0's momentum shard is saved;
        # accept it only when every buffer matches this rank's param sizes.
        opt_sd = state_dict.get("optimizer_state_dict")
        if opt_sd is not None and self._optimizer_state_matches(opt_sd):
            self.optimizer.load_state_dict(opt_sd)
            print_on_rank0("Successfully loaded optimizer state_dict.")
        else:
            print_on_rank0(
                "Optimizer momentum NOT restored (sharded FSDP state not restorable "
                "on this rank); Adam momentum reset on resume (re-warms in a few steps)."
            )
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
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
