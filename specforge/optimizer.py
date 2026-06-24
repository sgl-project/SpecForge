import torch

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        #   https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
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
        # Always restore the LR schedule (rank-agnostic).
        if "scheduler_state_dict" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            print_on_rank0("Successfully loaded scheduler state_dict.")
        opt_sd = state_dict.get("optimizer_state_dict")
        if opt_sd is not None and self._optimizer_state_matches(opt_sd):
            self.optimizer.load_state_dict(opt_sd)
            print_on_rank0("Successfully loaded optimizer state_dict.")
        else:
            print_on_rank0(
                "Optimizer momentum NOT restored (sharded FSDP state not restorable "
                "on this rank); Adam momentum reset on resume (re-warms in a few steps)."
            )

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
