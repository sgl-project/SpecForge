"""Shared optimizer registry, step-scalar checks, and the BF16 master adapter."""

from __future__ import annotations

import logging
import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.distributed as dist
from torch import nn

from specforge.lr_scheduler import build_lr_scheduler
from specforge.training.contracts import OptimizerSpec, SchedulerSpec
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


def local_tensor(tensor: Any) -> Any:
    """This rank's local shard of a DTensor, or ``tensor`` itself otherwise."""
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def like_parameter(value: Any, param: torch.Tensor) -> Any:
    """Wrap a saved rank-local tensor to match ``param``'s DTensor placement.

    Global shape and stride are taken from ``param`` so uneven shards restore
    identically. Scalars (e.g. AdamW's step tensor) are returned unchanged.
    """
    from torch.distributed.tensor import DTensor

    if not (isinstance(param, DTensor) and torch.is_tensor(value) and value.ndim):
        return value
    local = local_tensor(param)
    if value.shape != local.shape:
        raise ValueError(
            f"saved optimizer state of shape {tuple(value.shape)} does not match "
            f"this rank's shard {tuple(local.shape)}"
        )
    return DTensor.from_local(
        value.to(local.device),
        param.device_mesh,
        param.placements,
        run_check=False,
        shape=param.shape,
        stride=param.stride(),
    )


def _sum_of_squares(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """FP32 sum of squared L2 norms of ``tensors``.

    On CUDA one ``_foreach_norm`` launch per (device, dtype) group replaces
    three kernels per tensor; the result matches up to FP32 summation order.
    Grouping matters because a mixed-dtype list falls back to one kernel per
    tensor. Other devices keep the per-tensor reduction.
    """
    if all(tensor.is_cuda for tensor in tensors):
        groups: Dict[Tuple[torch.device, torch.dtype], List[torch.Tensor]] = {}
        for tensor in tensors:
            groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)
        norms = [
            norm
            for group in groups.values()
            for norm in torch._foreach_norm(group, 2.0, dtype=torch.float32)
        ]
        return torch.stack(norms).square().sum()
    return torch.stack([tensor.float().square().sum() for tensor in tensors]).sum()


def validate_step_scalars(
    grad_norm: torch.Tensor,
    *,
    max_grad_norm: float,
    loss_denominator: Optional[torch.Tensor] = None,
    extra_scalars: Sequence[torch.Tensor] = (),
    on_error: Callable[[], None],
) -> List[float]:
    """Validate global grad norm and optional loss denominator in one host read.

    Reads ``[grad_norm, *extra_scalars, *(loss_denominator)]`` in one stacked
    device-to-host transfer. Any invalid value invokes ``on_error`` (typically
    clearing accumulated gradients) before raising, so optimizer and scheduler
    state never advance past an invalid update.
    """
    checked = [grad_norm, *extra_scalars]
    if loss_denominator is not None:
        checked.append(loss_denominator)
    host_values = torch.stack(
        [
            value.detach().reshape(()).to(grad_norm.device, torch.float32)
            for value in checked
        ]
    ).tolist()
    if loss_denominator is not None:
        denominator = host_values[-1]
        if not math.isfinite(denominator) or denominator <= 0:
            on_error()
            raise ValueError("global loss denominator must be finite and positive")
    if not math.isfinite(host_values[0]):
        on_error()
        raise FloatingPointError(
            "refusing optimizer step with non-finite global grad norm "
            f"(max_grad_norm={max_grad_norm})"
        )
    return host_values


def restore_optimizer_kernel_state(
    optimizer: torch.optim.Optimizer, *, fused: Optional[bool] = None
) -> None:
    """Re-apply this run's AdamW fused/foreach flags after ``load_state_dict``.

    ``Optimizer.load_state_dict`` adopts saved param-group flags, so loading a
    checkpoint from an unfused run would silently disable ``fused`` here (and a
    fused checkpoint would break CPU masters). Fused AdamW requires step
    counters on the parameter's device; unfused keeps them on CPU.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        return
    if fused is None:
        params = [p for group in optimizer.param_groups for p in group["params"]]
        fused = (
            True if params and all(local_tensor(p).is_cuda for p in params) else None
        )
    for group in optimizer.param_groups:
        group["fused"] = fused
        if fused:
            group["foreach"] = None
    for param, state in optimizer.state.items():
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            state["step"] = (
                step.to(device=local_tensor(param).device, dtype=torch.float32)
                if fused
                else step.cpu()
            )


# ---------------------------------------------------------------------------
# Shared parameter-group policy registry
# ---------------------------------------------------------------------------


def _default_param_groups(
    named_params: Sequence[Tuple[str, nn.Parameter]],
    spec: OptimizerSpec,
    *,
    param_ndims: Optional[Mapping[int, int]] = None,
) -> List[Dict[str, Any]]:
    del spec, param_ndims
    return [{"params": [param for _, param in named_params]}]


def _fsdp1_param_ndims(module: nn.Module) -> Dict[int, int]:
    """Recover pre-flattening parameter ranks from any FSDP1 units in ``module``."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    ndims: Dict[int, int] = {}
    for unit in module.modules():
        if isinstance(unit, FSDP) and unit._handle is not None:
            flat = unit._handle.flat_param
            for param, shape in zip(flat._params, flat._shapes):
                ndims[id(param)] = len(shape)
    return ndims


def _no_decay_1d_param_groups(
    named_params: Sequence[Tuple[str, nn.Parameter]],
    spec: OptimizerSpec,
    *,
    param_ndims: Optional[Mapping[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Apply ``weight_decay`` only to >=2-D non-bias tensors."""
    decay, no_decay = [], []
    for name, param in named_params:
        ndim = (
            param_ndims.get(id(param), param.ndim)
            if param_ndims is not None
            else param.ndim
        )
        if ndim >= 2 and not name.endswith(".bias") and name != "bias":
            decay.append(param)
        else:
            no_decay.append(param)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": spec.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


_PARAM_GROUP_POLICIES: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "default": _default_param_groups,
    "no_decay_1d": _no_decay_1d_param_groups,
}


def register_param_group_policy(
    name: str,
    policy: Callable[..., List[Dict[str, Any]]],
    *,
    override: bool = False,
) -> None:
    """Register a parameter-grouping policy under ``name``.

    A policy is called as ``policy(named_params, spec, *, param_ndims)``.
    ``param_ndims`` maps ``id(param)`` to the parameter's rank before a
    sharding wrapper flattened it, or is None when nothing was flattened.
    """
    if not name:
        raise ValueError("param_group_policy name must be non-empty")
    if name in _PARAM_GROUP_POLICIES and not override:
        raise ValueError(f"param_group_policy {name!r} is already registered")
    _PARAM_GROUP_POLICIES[name] = policy


def registered_param_group_policies() -> Tuple[str, ...]:
    """Registered parameter-group policy names, sorted."""
    return tuple(sorted(_PARAM_GROUP_POLICIES))


def build_param_groups(
    named_params: Iterable[Tuple[str, nn.Parameter]],
    spec: OptimizerSpec,
    *,
    param_ndims: Optional[Mapping[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Group trainable ``(name, parameter)`` pairs using ``spec.param_group_policy``.

    ``param_ndims`` maps ``id(param)`` to the parameter's pre-sharding rank
    when a wrapper such as FSDP1 has flattened shards to 1-D.
    """
    policy = _PARAM_GROUP_POLICIES.get(spec.param_group_policy)
    if policy is None:
        raise ValueError(
            f"unsupported param_group_policy={spec.param_group_policy!r}; "
            f"expected one of {sorted(_PARAM_GROUP_POLICIES)}"
        )
    trainable = [(name, p) for name, p in named_params if p.requires_grad]
    if not trainable:
        raise ValueError("the optimizer's target has no trainable parameter")
    return policy(trainable, spec, param_ndims=param_ndims)


# ---------------------------------------------------------------------------
# Shared optimizer registry
# ---------------------------------------------------------------------------


def _build_adamw(
    param_groups: List[Dict[str, Any]], spec: OptimizerSpec
) -> torch.optim.Optimizer:
    params = [p for group in param_groups for p in group["params"]]
    # One fused kernel updates every CUDA parameter; CPU parameters keep the
    # default AdamW implementation.
    fused = True if params and all(local_tensor(p).is_cuda for p in params) else None
    return torch.optim.AdamW(
        param_groups,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        betas=spec.betas,
        eps=spec.eps,
        fused=fused,
    )


_OPTIMIZERS: Dict[
    str, Callable[[List[Dict[str, Any]], OptimizerSpec], torch.optim.Optimizer]
] = {
    "adamw": _build_adamw,
}


def register_optimizer(
    name: str,
    builder: Callable[[List[Dict[str, Any]], OptimizerSpec], torch.optim.Optimizer],
    *,
    override: bool = False,
) -> None:
    """Register an optimizer builder under ``name``."""
    if not name:
        raise ValueError("optimizer name must be non-empty")
    if name in _OPTIMIZERS and not override:
        raise ValueError(f"optimizer {name!r} is already registered")
    _OPTIMIZERS[name] = builder


def registered_optimizers() -> Tuple[str, ...]:
    """Registered optimizer names, sorted."""
    return tuple(sorted(_OPTIMIZERS))


def validate_optimizer_spec(spec: OptimizerSpec) -> None:
    """Validate optimizer and parameter-group policy names and built-in options."""
    if spec.kind not in _OPTIMIZERS:
        raise ValueError(
            f"unsupported optimizer={spec.kind!r}; expected one of {sorted(_OPTIMIZERS)}"
        )
    if spec.param_group_policy not in _PARAM_GROUP_POLICIES:
        raise ValueError(
            f"unsupported param_group_policy={spec.param_group_policy!r}; "
            f"expected one of {sorted(_PARAM_GROUP_POLICIES)}"
        )
    if spec.kind == "adamw" and spec.extra:
        raise ValueError(
            f"adamw optimizer accepts no extra options, got {sorted(spec.extra)}"
        )


def optimizer_resume_metadata(
    *,
    optimizer_spec: Optional[OptimizerSpec] = None,
    scheduler_spec: Optional[SchedulerSpec] = None,
) -> Dict[str, Any]:
    """Build checkpoint resume metadata for optimizer/scheduler configuration."""
    meta: Dict[str, Any] = {}
    if optimizer_spec is not None:
        meta["optimizer_type"] = optimizer_spec.kind
        meta["param_group_policy"] = optimizer_spec.param_group_policy
        meta["optimizer_options"] = dict(optimizer_spec.extra)
    if scheduler_spec is not None:
        meta["lr_scheduler_type"] = scheduler_spec.kind
        meta["lr_scheduler_min_lr_ratio"] = float(scheduler_spec.min_lr_ratio)
        meta["lr_scheduler_options"] = dict(scheduler_spec.extra)
    return meta


def validate_optimizer_resume_metadata(
    state_dict: Mapping[str, Any],
    *,
    optimizer_spec: Optional[OptimizerSpec] = None,
    scheduler_spec: Optional[SchedulerSpec] = None,
    default_scheduler_type: Optional[str] = "cosine",
) -> None:
    """Refuse resuming across incompatible optimizer or LR-scheduler settings.

    Missing keys fall back to pre-registry legacy defaults (``adamw``,
    ``default`` policy, ``cosine`` scheduler with ``min_lr_ratio=0.0`` and no
    extra options) so legacy checkpoints remain resumable while catching any
    configuration drift—including switching ``cosine`` between
    ``min_lr_ratio == 0.0`` (``CosineAnnealingWarmupLR``) and ``> 0.0``
    (``FunctionalLRScheduler``).
    """
    if optimizer_spec is not None:
        saved_opt = state_dict.get("optimizer_type", "adamw")
        if saved_opt != optimizer_spec.kind:
            raise ValueError(
                f"checkpoint optimizer used optimizer={saved_opt!r} but this run "
                f"has optimizer={optimizer_spec.kind!r}"
            )
        saved_policy = state_dict.get("param_group_policy", "default")
        if saved_policy != optimizer_spec.param_group_policy:
            raise ValueError(
                "checkpoint optimizer used param_group_policy="
                f"{saved_policy!r} but this run has "
                f"param_group_policy={optimizer_spec.param_group_policy!r}"
            )
        saved_opt_extra = dict(state_dict.get("optimizer_options") or {})
        live_opt_extra = dict(optimizer_spec.extra)
        if saved_opt_extra != live_opt_extra:
            raise ValueError(
                "checkpoint optimizer used optimizer_options="
                f"{saved_opt_extra!r} but this run has "
                f"optimizer_options={live_opt_extra!r}"
            )
    if scheduler_spec is not None:
        saved_sched = state_dict.get("lr_scheduler_type", default_scheduler_type)
        if saved_sched != scheduler_spec.kind:
            raise ValueError(
                "checkpoint optimizer used lr_scheduler="
                f"{saved_sched!r} but this run has "
                f"lr_scheduler={scheduler_spec.kind!r}"
            )
        saved_min_lr = float(state_dict.get("lr_scheduler_min_lr_ratio", 0.0))
        if saved_min_lr != float(scheduler_spec.min_lr_ratio):
            raise ValueError(
                "checkpoint optimizer used lr_scheduler_min_lr_ratio="
                f"{saved_min_lr!r} but this run has "
                f"lr_scheduler_min_lr_ratio={float(scheduler_spec.min_lr_ratio)!r}"
            )
        saved_sched_extra = dict(state_dict.get("lr_scheduler_options") or {})
        live_sched_extra = dict(scheduler_spec.extra)
        if saved_sched_extra != live_sched_extra:
            raise ValueError(
                "checkpoint optimizer used lr_scheduler_options="
                f"{saved_sched_extra!r} but this run has "
                f"lr_scheduler_options={live_sched_extra!r}"
            )


def build_optimizer(
    target: nn.Module | Iterable[Tuple[str, nn.Parameter]],
    spec: OptimizerSpec,
    *,
    param_ndims: Optional[Mapping[int, int]] = None,
) -> torch.optim.Optimizer:
    """Build an optimizer from the shared registry over ``target``."""
    validate_optimizer_spec(spec)
    named_params = (
        target.named_parameters() if isinstance(target, nn.Module) else target
    )
    param_groups = build_param_groups(named_params, spec, param_ndims=param_ndims)
    return _OPTIMIZERS[spec.kind](param_groups, spec)


# ---------------------------------------------------------------------------
# Legacy FSDP1/DDP FP32 master-copy adapter
# ---------------------------------------------------------------------------


class BF16Optimizer:
    """FP32 master-copy adapter over the shared optimizer and LR registries.

    Used by the legacy ``fsdp`` runtime where trainable parameters live in
    ``model.torch_dtype`` (typically ``bf16``) and require separate ``fp32``
    master copies. Under ``fsdp2`` the parameters themselves are ``fp32``
    masters, so the runtime drives the registry-built optimizer directly.
    """

    #: ``step(loss_denominator=...)`` validates the caller's global loss
    #: denominator in the same host read as the grad norm (see the step runner).
    checks_loss_denominator = True

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.5,
        total_steps: int = 800_000,
        warmup_ratio: float = 0.015,
        lr_scheduler: str = "cosine",
        offload_master: bool = False,
        *,
        optimizer_spec: Optional[OptimizerSpec] = None,
        scheduler_spec: Optional[SchedulerSpec] = None,
        param_ndims: Optional[Mapping[int, int]] = None,
    ):
        opt_spec = (
            optimizer_spec
            if optimizer_spec is not None
            else OptimizerSpec(
                kind="adamw",
                lr=lr,
                weight_decay=weight_decay,
                offload_master=bool(offload_master),
            )
        )
        sched_spec = (
            scheduler_spec
            if scheduler_spec is not None
            else SchedulerSpec(
                kind=lr_scheduler,
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
            )
        )
        self._optimizer_spec = opt_spec
        self._scheduler_spec = sched_spec
        self.model = model
        named_trainable = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        self.model_params = [p for _, p in named_trainable]
        self.max_grad_norm = max_grad_norm
        self.offload_master = bool(opt_spec.offload_master)
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
        named_masters = [
            (name, mp) for (name, _), mp in zip(named_trainable, self.fp32_params)
        ]
        resolved_ndims = _fsdp1_param_ndims(model)
        if param_ndims is not None:
            resolved_ndims.update(param_ndims)
        master_ndims = (
            {
                id(mp): resolved_ndims.get(id(p), p.ndim)
                for p, mp in zip(self.model_params, self.fp32_params)
            }
            if resolved_ndims
            else None
        )
        self._adamw_fused = (
            True
            if self.fp32_params and all(mp.is_cuda for mp in self.fp32_params)
            else None
        )
        self.optimizer = build_optimizer(
            named_masters, opt_spec, param_ndims=master_ndims
        )
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        self.lr_scheduler_type = sched_spec.kind
        self.scheduler = build_lr_scheduler(self.optimizer, sched_spec)

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
        Either check fails before optimizer, scheduler, or copy-back state changes.
        """
        grad_norm, clip_coefficient = self._grad_norm_and_clip_coefficient()
        extra = (clip_coefficient,) if self.offload_master else ()
        try:
            host_values = validate_step_scalars(
                grad_norm,
                max_grad_norm=self.max_grad_norm,
                loss_denominator=loss_denominator,
                extra_scalars=extra,
                on_error=self._clear_grads,
            )
        except FloatingPointError:
            self.last_grad_norm = grad_norm.detach()
            raise
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
        restore_optimizer_kernel_state(self.optimizer, fused=self._adamw_fused)

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        validate_optimizer_resume_metadata(
            state_dict,
            optimizer_spec=self._optimizer_spec,
            scheduler_spec=self._scheduler_spec,
            default_scheduler_type="cosine",
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

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            **optimizer_resume_metadata(
                optimizer_spec=self._optimizer_spec,
                scheduler_spec=self._scheduler_spec,
            ),
            "max_grad_norm": self.max_grad_norm,
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]


def master_dtype_of(optimizer) -> Optional[torch.dtype]:
    """The dtype ``optimizer`` keeps its own copy of the parameters in.

    ``torch.float32`` for ``BF16Optimizer`` (separate master copies); ``None``
    for any standard ``torch.optim.Optimizer``, which updates the parameters it
    was given directly.
    """
    if isinstance(optimizer, BF16Optimizer):
        return torch.float32
    if isinstance(optimizer, torch.optim.Optimizer):
        return None
    raise TypeError(f"unknown optimizer {type(optimizer).__name__}")


__all__ = [
    "BF16Optimizer",
    "build_optimizer",
    "build_param_groups",
    "like_parameter",
    "local_tensor",
    "master_dtype_of",
    "optimizer_resume_metadata",
    "register_optimizer",
    "register_param_group_policy",
    "registered_optimizers",
    "registered_param_group_policies",
    "restore_optimizer_kernel_state",
    "validate_optimizer_resume_metadata",
    "validate_optimizer_spec",
    "validate_step_scalars",
]
