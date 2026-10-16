import logging
import math

import torch
import torch.distributed as dist

from specforge.lr_scheduler import ConstantWarmupLR, CosineAnnealingWarmupLR
from specforge.muon import (
    ADAMW,
    MUON,
    MUON_GROUP_RULE,
    OPTIMIZER_KINDS,
    STATE_FORMAT,
    MuonWithAuxAdamW,
    ShardedMuon,
    canonical_parameter_name,
    capture_parameter_layout,
)
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


def _sum_of_squares(tensors):
    """FP32 sum of squared L2 norms of ``tensors``.

    On CUDA one ``_foreach_norm`` launch per (device, dtype) group replaces
    three kernels per tensor; the result matches up to FP32 summation order.
    Grouping matters because a mixed-dtype list falls back to one kernel per
    tensor. Other devices keep the per-tensor reduction.
    """
    if all(tensor.is_cuda for tensor in tensors):
        groups = {}
        for tensor in tensors:
            groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)
        norms = [
            norm
            for group in groups.values()
            for norm in torch._foreach_norm(group, 2.0, dtype=torch.float32)
        ]
        return torch.stack(norms).square().sum()
    return torch.stack([tensor.float().square().sum() for tensor in tensors]).sum()


def _plain(value):
    """Tuples -> lists, recursively (JSON shape; comparisons across torch.save)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class BF16Optimizer:
    """AdamW over fp32 master copies of the bf16 trainable params, with grad
    clipping and configurable warmup scheduling.

    ``optimizer="muon"`` (opt-in; see :mod:`specforge.muon`) runs Muon on the
    draft's decoder-layer matrices and the same AdamW on every other master,
    under the one global clip and one LR schedule (separate peak rates ``lr``
    and ``muon_lr``). The default ``"adamw"`` path and checkpoint layout are
    unchanged.
    """

    #: ``step(loss_denominator=...)`` validates the caller's global loss
    #: denominator in the same host read as the grad norm (see TrainerCore).
    checks_loss_denominator = True

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
        *,
        optimizer=ADAMW,
        muon_lr=None,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        muon_weight_decay=0.0,
        muon_adjust_lr_fn="original",
        parameter_layout=None,
    ):
        if optimizer not in OPTIMIZER_KINDS:
            raise ValueError(
                f"unsupported optimizer={optimizer!r}; "
                f"expected one of {list(OPTIMIZER_KINDS)}"
            )
        if optimizer == MUON and offload_master:
            raise ValueError("optimizer='muon' does not support offload_master")
        self.optimizer_kind = optimizer
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
        # One fused kernel updates every CUDA master; CPU-offloaded masters keep
        # the default AdamW implementation.
        self._adamw_fused = (
            True
            if self.fp32_params and all(mp.is_cuda for mp in self.fp32_params)
            else None
        )
        if optimizer == ADAMW:
            self.optimizer = torch.optim.AdamW(
                self.fp32_params,
                lr=lr,
                weight_decay=weight_decay,
                fused=self._adamw_fused,
            )
            self.adamw_optimizer = self.optimizer
        else:
            self.optimizer = self._build_muon(
                model,
                parameter_layout,
                lr=lr,
                weight_decay=weight_decay,
                muon_lr=muon_lr,
                muon_momentum=muon_momentum,
                muon_nesterov=muon_nesterov,
                muon_ns_steps=muon_ns_steps,
                muon_weight_decay=muon_weight_decay,
                muon_adjust_lr_fn=muon_adjust_lr_fn,
            )
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

    def _build_muon(
        self,
        model,
        parameter_layout,
        *,
        lr,
        weight_decay,
        muon_lr,
        muon_momentum,
        muon_nesterov,
        muon_ns_steps,
        muon_weight_decay,
        muon_adjust_lr_fn,
    ):
        """Split the fp32 masters into the Muon and AdamW groups.

        ``parameter_layout`` (canonical name -> logical shape / group) must be
        captured before FSDP wraps the model; without it the model must still
        hold its original (unflattened) parameters.
        """
        if muon_lr is None or not muon_lr > 0:
            raise ValueError(
                f"optimizer='muon' requires a positive muon_lr, got {muon_lr}"
            )
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if [id(p) for _, p in named] != [id(p) for p in self.model_params]:
            raise RuntimeError(
                "named trainable parameters differ from model.parameters()"
            )
        names = [canonical_parameter_name(n) for n, _ in named]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate canonical parameter names: {names}")
        layout = (
            capture_parameter_layout(model)
            if parameter_layout is None
            else dict(parameter_layout)
        )
        if set(layout) != set(names):
            raise ValueError(
                "the Muon parameter layout does not match the trainable parameters: "
                f"missing={sorted(set(names) - set(layout))}, "
                f"unexpected={sorted(set(layout) - set(names))}"
            )
        for name, master in zip(names, self.fp32_params):
            info = layout[name]
            if master.numel() > math.prod(info.shape):
                raise ValueError(
                    f"{name}: {master.numel()} local elements exceed the logical "
                    f"shape {info.shape}"
                )
        muon_index = [i for i, n in enumerate(names) if layout[n].muon]
        adamw_index = [i for i, n in enumerate(names) if not layout[n].muon]
        if not muon_index:
            raise ValueError(f"optimizer='muon' found no {MUON_GROUP_RULE} matrix")
        if not adamw_index:
            raise ValueError("optimizer='muon' expects the AdamW group to be non-empty")
        self._parameter_names = tuple(names)
        self._parameter_layout = {n: layout[n] for n in names}
        self.adamw_optimizer = torch.optim.AdamW(
            [self.fp32_params[i] for i in adamw_index],
            lr=lr,
            weight_decay=weight_decay,
            fused=self._adamw_fused,
        )
        self.muon_optimizer = ShardedMuon(
            [self.fp32_params[i] for i in muon_index],
            names=[names[i] for i in muon_index],
            shapes=[layout[names[i]].shape for i in muon_index],
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            weight_decay=muon_weight_decay,
            adjust_lr_fn=muon_adjust_lr_fn,
        )
        self.muon_hyperparameters = {
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "muon_learning_rate": float(muon_lr),
            "muon_momentum": float(muon_momentum),
            "muon_nesterov": bool(muon_nesterov),
            "muon_ns_steps": int(muon_ns_steps),
            "muon_weight_decay": float(muon_weight_decay),
            "muon_adjust_lr_fn": muon_adjust_lr_fn,
            "muon_ns_coefficients": list(
                self.muon_optimizer.defaults["ns_coefficients"]
            ),
            "muon_eps": float(self.muon_optimizer.defaults["eps"]),
            "group_rule": MUON_GROUP_RULE,
        }
        membership = self.muon_membership()
        lines = [
            f"Muon optimizer: {len(muon_index)} matrices "
            f"({membership['muon_numel']:,} params) on Muon, {len(adamw_index)} "
            f"tensors ({membership['adamw_numel']:,} params) on AdamW; "
            f"muon_lr={muon_lr} adamw_lr={lr} momentum={muon_momentum} "
            f"nesterov={muon_nesterov} ns_steps={muon_ns_steps} "
            f"muon_weight_decay={muon_weight_decay} "
            f"adjust_lr_fn={muon_adjust_lr_fn}"
        ]
        for group_name in (MUON, ADAMW):
            lines.extend(
                f"  {group_name}: {name} {list(shape)}"
                for name, shape in membership[group_name]
            )
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            logger.info("\n".join(lines))
        return MuonWithAuxAdamW(self.adamw_optimizer, self.muon_optimizer)

    def muon_membership(self) -> dict:
        """Muon / AdamW membership (canonical names, logical shapes; model order)."""
        if self.optimizer_kind != MUON:
            raise RuntimeError("muon_membership applies to optimizer='muon' only")
        groups = {MUON: [], ADAMW: []}
        for name in self._parameter_names:
            info = self._parameter_layout[name]
            groups[MUON if info.muon else ADAMW].append((name, tuple(info.shape)))
        return {
            **groups,
            "muon_numel": sum(math.prod(shape) for _, shape in groups[MUON]),
            "adamw_numel": sum(math.prod(shape) for _, shape in groups[ADAMW]),
        }

    def _muon_record(self) -> dict:
        """What a Muon checkpoint pins: membership, shapes and hyperparameters."""
        return {
            "format": STATE_FORMAT,
            "parameters": [
                [
                    name,
                    MUON if self._parameter_layout[name].muon else ADAMW,
                    list(self._parameter_layout[name].shape),
                ]
                for name in self._parameter_names
            ],
            "hyperparameters": dict(self.muon_hyperparameters),
        }

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        Muon binds the same group: it owns the disjoint matrix slices.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled
        if self.optimizer_kind == MUON:
            self.muon_optimizer.configure_sharding(
                process_group=process_group, enabled=enabled
            )

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
            total_norm_sq = _sum_of_squares(grads)
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

    def _clear_grads(self) -> None:
        with torch.no_grad():
            for p in self.model_params:
                p.grad = None
            for mp in self.fp32_params:
                mp.grad = None

    def step(self, *, loss_denominator=None):
        """Clip, update, and return the pre-clip global grad norm.

        The step synchronizes with the host once: the grad norm (plus the CPU
        clip coefficient when masters are offloaded) and the optional
        ``loss_denominator`` -- the all-reduced global loss denominator the
        caller already scaled the gradients by -- are read in one transfer.
        Either check fails before Adam, scheduler, or copy-back state changes.
        """
        grad_norm, clip_coefficient = self._grad_norm_and_clip_coefficient()
        checked = [grad_norm]
        if self.offload_master:
            checked.append(clip_coefficient)
        if loss_denominator is not None:
            checked.append(loss_denominator)
        # FP32 is exact for the norm and clip coefficient; the denominator
        # check needs only its sign and finiteness.
        host_values = torch.stack(
            [
                value.detach().reshape(()).to(grad_norm.device, torch.float32)
                for value in checked
            ]
        ).tolist()
        if loss_denominator is not None:
            denominator = host_values[-1]
            if not math.isfinite(denominator) or denominator <= 0:
                # Gradients were already scaled by an invalid factor.
                self._clear_grads()
                raise ValueError("global loss denominator must be finite and positive")
        if not math.isfinite(host_values[0]):
            # The norm is already all-reduced, so every rank fails before Adam,
            # scheduler, global-step, or durable-ack state can advance. Returning
            # here would make the controller record an optimizer update that did
            # not happen and permanently discard its training window.
            self._clear_grads()
            self.last_grad_norm = grad_norm.detach()
            raise FloatingPointError(
                "refusing optimizer step with non-finite global grad norm "
                f"(max_grad_norm={self.max_grad_norm})"
            )
        with torch.no_grad():
            model_grads, master_grads = [], []
            for p, mp in zip(self.model_params, self.fp32_params):
                if p.grad is None:
                    mp.grad = None
                    continue
                if self.offload_master:
                    master_grad = p.grad.detach().to(
                        device=mp.device,
                        dtype=torch.float32,
                    )
                    master_grad.mul_(host_values[1])
                else:
                    master_grad = torch.empty_like(mp)
                    model_grads.append(p.grad.detach())
                    master_grads.append(master_grad)
                mp.grad = master_grad
            if master_grads:
                torch._foreach_copy_(master_grads, model_grads)
                torch._foreach_mul_(master_grads, clip_coefficient)
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            if self.offload_master:
                for p, mp in zip(self.model_params, self.fp32_params):
                    p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
            elif self.model_params:
                torch._foreach_copy_(
                    [p.data for p in self.model_params],
                    [mp.data for mp in self.fp32_params],
                )
            for p in self.model_params:
                p.grad = None
        return self.last_grad_norm

    def _restore_adamw_implementation(self) -> None:
        """Re-apply this run's AdamW kernel choice after a checkpoint load.

        ``Optimizer.load_state_dict`` adopts the saved param-group flags, so a
        checkpoint from an unfused run would silently disable ``fused`` here
        (and a fused one would enable it for CPU masters). Fused AdamW reads
        its step counters on the parameter device; unfused keeps them on CPU.
        """
        for group in self.adamw_optimizer.param_groups:
            group["fused"] = self._adamw_fused
            if self._adamw_fused:
                group["foreach"] = None
        for param, state in self.adamw_optimizer.state.items():
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                state["step"] = (
                    step.to(device=param.device, dtype=torch.float32)
                    if self._adamw_fused
                    else step.cpu()
                )

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        self._check_optimizer_kind_resume(state_dict)
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
        self._restore_adamw_implementation()
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

    def _check_optimizer_kind_resume(self, state_dict):
        """Fail closed on an optimizer change; a Muon resume must also keep the
        exact membership, logical shapes and hyperparameters."""
        if "optimizer_type" in state_dict:
            raise ValueError(
                "checkpoint optimizer uses the retired draft hybrid-Muon format; "
                "start a new optimizer state"
            )
        saved_kind = state_dict.get("optimizer_kind", ADAMW)
        if saved_kind != self.optimizer_kind:
            raise ValueError(
                f"checkpoint optimizer is {saved_kind!r} but this run has "
                f"optimizer={self.optimizer_kind!r}; resume with the original optimizer"
            )
        if self.optimizer_kind != MUON:
            return
        saved = state_dict.get("muon")
        current = self._muon_record()
        if not isinstance(saved, dict) or saved.get("format") != STATE_FORMAT:
            raise ValueError(
                "checkpoint optimizer is 'muon' but carries no Muon record of format "
                f"{STATE_FORMAT}"
            )
        if _plain(saved.get("parameters")) != _plain(current["parameters"]):
            raise ValueError(
                "checkpoint Muon/AdamW parameter membership or logical shapes differ "
                "from this run's"
            )
        if _plain(saved.get("hyperparameters")) != _plain(current["hyperparameters"]):
            raise ValueError(
                "checkpoint Muon hyperparameters "
                f"{_plain(saved.get('hyperparameters'))} differ from this run's "
                f"{_plain(current['hyperparameters'])}"
            )

    def state_dict(self):
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }
        if self.optimizer_kind == MUON:
            # Only Muon adds keys; AdamW keeps the historical layout (whose
            # absence of ``optimizer_kind`` means AdamW on resume).
            state["optimizer_kind"] = MUON
            state["muon"] = self._muon_record()
        return state

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def learning_rates_by_group(self) -> dict:
        """``{"adamw": lr, "muon": lr}`` for Muon runs (logged as ``lr_<group>``);
        empty for AdamW, whose only rate is :meth:`get_learning_rate`."""
        if self.optimizer_kind != MUON:
            return {}
        adamw_group, muon_group = self.optimizer.param_groups
        return {ADAMW: adamw_group["lr"], MUON: muon_group["lr"]}
