"""Muon on the draft's decoder-layer matrices, AdamW on the rest.

``training.optimizer: muon`` replaces AdamW on every 2-D ``nn.Linear`` weight
inside the draft's decoder layers (``layers.<i>.*`` or ``midlayer.*``:
attention q/k/v/o and MLP gate/up/down). Every other trainable tensor (``fc``,
embedding projections, norms, biases, GRUs, selector/Markov/confidence heads,
any trainable embedding or LM head) stays on the AdamW of
:class:`specforge.optimizer.BF16Optimizer`.

The default update is the ORIGINAL Muon (Keller Jordan; ``torch.optim.Muon``
with ``adjust_lr_fn='original'``). For a matrix ``W`` of logical shape
``rows x cols`` and its clipped fp32 master gradient ``g``::

    buf = momentum * buf + (1 - momentum) * g        # buf.lerp_(g, 1 - momentum)
    u   = g.lerp(buf, momentum) if nesterov else buf
    O   = NewtonSchulz5(u)                           # bf16, quintic (3.4445, -4.7750, 2.0315)
    W  -= lr_t * sqrt(max(1, rows / cols)) * O       # (decoupled weight decay first, default 0)

``match_rms_adamw`` replaces ``sqrt(max(1, rows / cols))`` with
``0.2 * sqrt(max(rows, cols))``; its learning rates are on AdamW's scale.

Under FSDP1 with ``use_orig_params`` each rank's optimizer sees a flat,
possibly empty, contiguous slice of every matrix (rank-ordered). The momentum
stays sharded; per update one ``all_to_all`` sends every rank's bf16 slice of
``u`` to the matrix's owner rank (owners are balanced once by Newton-Schulz
cost), the owner orthogonalizes the full logical matrix -- the same ops on the
same bytes as the unsharded single-GPU computation -- and a second
``all_to_all`` returns each rank its slice of ``O``. Every rank therefore
applies values that are identical to the unsharded update, and the work is
spread across the shard group instead of repeated on every rank.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

ADAMW = "adamw"
MUON = "muon"
OPTIMIZER_KINDS = (ADAMW, MUON)
#: The Muon group rule (recorded in the startup log and the checkpoint).
MUON_GROUP_RULE = "2-D nn.Linear.weight under layers.<i>. or midlayer."
MUON_LR_ADJUSTMENTS = ("original", "match_rms_adamw")
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-7
STATE_FORMAT = 1

_WRAPPER_SEGMENTS = frozenset(
    {"_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"}
)
_DECODER_LAYER_RE = re.compile(r"(?:^|\.)(?:layers\.\d+|midlayer)\.")


def canonical_parameter_name(name: str) -> str:
    """Drop FSDP / activation-checkpoint / compile wrapper segments."""

    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


@dataclass(frozen=True)
class ParameterInfo:
    """Logical (pre-FSDP) shape of one trainable tensor and its group."""

    shape: Tuple[int, ...]
    muon: bool


def capture_parameter_layout(model: nn.Module) -> Dict[str, ParameterInfo]:
    """Canonical name -> :class:`ParameterInfo` for every trainable tensor.

    Must run before FSDP wraps ``model``: ``use_orig_params`` turns every
    original parameter into a flat 1-D view of the local shard. A tensor joins
    the Muon group iff it is 2-D and every module that owns it is an
    ``nn.Linear`` whose ``weight`` it is, inside a decoder layer.
    """

    owners: Dict[int, List[Tuple[str, nn.Module, str]]] = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        for leaf, parameter in module.named_parameters(recurse=False):
            full = f"{module_name}.{leaf}" if module_name else leaf
            owners.setdefault(id(parameter), []).append(
                (canonical_parameter_name(full), module, leaf)
            )
    layout: Dict[str, ParameterInfo] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        linear_weights = [
            isinstance(module, nn.Linear) and leaf == "weight"
            for _, module, leaf in owners[id(parameter)]
        ]
        if any(linear_weights) and parameter.ndim != 2:
            raise ValueError(
                f"{name} is an nn.Linear weight of shape {tuple(parameter.shape)}: "
                "capture the Muon parameter layout before FSDP flattens the model"
            )
        # Check every owner: a layer projection may share an embedding/head weight.
        use_muon = parameter.ndim == 2 and all(
            is_weight and _DECODER_LAYER_RE.search(owner)
            for (owner, _, _), is_weight in zip(owners[id(parameter)], linear_weights)
        )
        layout[canonical_parameter_name(name)] = ParameterInfo(
            shape=tuple(int(size) for size in parameter.shape), muon=bool(use_muon)
        )
    return layout


def muon_lr_ratio(shape: Sequence[int], adjust_lr_fn: str = "original") -> float:
    """``torch.optim.Muon``'s shape-aware learning-rate multiplier."""

    rows, cols = shape[:2]
    if adjust_lr_fn == "original":
        return math.sqrt(max(1, rows / cols))
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, cols))
    raise ValueError(
        f"unsupported Muon adjust_lr_fn={adjust_lr_fn!r}; "
        f"expected one of {list(MUON_LR_ADJUSTMENTS)}"
    )


def zeropower_via_newtonschulz5(
    update: torch.Tensor,
    *,
    ns_steps: int,
    coefficients: Sequence[float] = NS_COEFFICIENTS,
    eps: float = NS_EPS,
) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization, op for op ``torch.optim.Muon``'s.

    bf16; the iteration runs on the transposed (wide) side when rows > cols
    and the input is first scaled by its Frobenius norm (``clamp(min=eps)``).
    A bf16 ``update`` is normalized in place (callers pass a scratch buffer).
    """

    if update.ndim != 2:
        raise ValueError(f"Newton-Schulz needs a 2-D matrix, got {tuple(update.shape)}")
    if not 0 < int(ns_steps) < 100:
        raise ValueError(f"ns_steps must be in [1, 99], got {ns_steps}")
    a, b, c = coefficients
    ortho = update.bfloat16()
    transposed = update.size(0) > update.size(1)
    if transposed:
        ortho = ortho.T
    ortho.div_(ortho.norm().clamp(min=eps))
    for _ in range(int(ns_steps)):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.T
    return ortho


def newton_schulz_cost(shape: Sequence[int]) -> int:
    """Multiply-adds of one Newton-Schulz step: 2 m^2 n + m^3 (m <= n)."""

    m, n = sorted(int(size) for size in shape[:2])
    return m * m * (2 * n + m)


def assign_owners(shapes: Sequence[Sequence[int]], world: int) -> Tuple[int, ...]:
    """Deterministic LPT assignment of matrices to ranks by Newton-Schulz cost."""

    loads = [0] * world
    owners = [0] * len(shapes)
    order = sorted(
        range(len(shapes)), key=lambda i: (-newton_schulz_cost(shapes[i]), i)
    )
    for index in order:
        rank = min(range(world), key=lambda r: (loads[r], r))
        owners[index] = rank
        loads[rank] += newton_schulz_cost(shapes[index])
    return tuple(owners)


class _MuonPlan:
    """Where every rank's slice of every Muon matrix lives and who owns it.

    ``sizes[r][i]`` / ``offsets[r][i]``: rank ``r``'s slice of matrix ``i`` in the
    row-major flattened matrix. Local mode (one rank, full matrices) runs
    without collectives.
    """

    def __init__(self, shapes, sizes, owners, *, rank, group=None):
        self.shapes = tuple(tuple(shape) for shape in shapes)
        self.sizes = tuple(tuple(row) for row in sizes)
        self.world = len(self.sizes)
        self.rank = int(rank)
        self.group = group
        self.owners = tuple(owners)
        count = len(self.shapes)
        for index, shape in enumerate(self.shapes):
            total = sum(row[index] for row in self.sizes)
            if total != math.prod(shape):
                raise RuntimeError(
                    f"Muon matrix {index} of logical shape {shape}: the rank slices "
                    f"{[row[index] for row in self.sizes]} do not reconstruct it"
                )
        self.offsets = tuple(
            tuple(sum(self.sizes[s][i] for s in range(r)) for i in range(count))
            for r in range(self.world)
        )
        self.holders = tuple(
            sum(1 for r in range(self.world) if self.sizes[r][i]) for i in range(count)
        )
        me = self.rank
        self.owned = tuple(
            tuple(i for i in range(count) if self.owners[i] == r)
            for r in range(self.world)
        )
        # Send layout: per destination owner, its matrices in index order.
        self.send_pos: Dict[int, int] = {}
        self.send_splits: List[int] = []
        position = 0
        for destination in range(self.world):
            start = position
            for index in self.owned[destination]:
                self.send_pos[index] = position
                position += self.sizes[me][index]
            self.send_splits.append(position - start)
        self.send_total = position
        # Receive layout: per source rank, my matrices in index order.
        self.recv_pos: Dict[Tuple[int, int], int] = {}
        self.recv_splits: List[int] = []
        position = 0
        for source in range(self.world):
            start = position
            for index in self.owned[me]:
                self.recv_pos[(source, index)] = position
                position += self.sizes[source][index]
            self.recv_splits.append(position - start)
        self.recv_total = position

    @property
    def sharded(self) -> bool:
        return self.world > 1

    @classmethod
    def local(cls, shapes):
        return cls(
            shapes, [[math.prod(shape) for shape in shapes]], [0] * len(shapes), rank=0
        )

    def describe(self) -> dict:
        return {
            "mode": "sharded" if self.sharded else "local",
            "world": self.world,
            "rank": self.rank,
            "owners": list(self.owners),
            "local_numels": list(self.sizes[self.rank]),
            "offsets": list(self.offsets[self.rank]),
            "owned_cost": [
                sum(newton_schulz_cost(self.shapes[i]) for i in self.owned[r])
                for r in range(self.world)
            ],
        }

    def active(self, params: Sequence[torch.Tensor]) -> List[bool]:
        """Matrices with a gradient: every slice holder must agree (torch skips
        a parameter without one)."""

        local = [bool(p.numel()) and p.grad is not None for p in params]
        if not self.sharded:
            return local
        device = params[0].device
        flags = torch.tensor(local, dtype=torch.int32, device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.SUM, group=self.group)
        counts = flags.tolist()
        active = []
        for index, (count, holders) in enumerate(zip(counts, self.holders)):
            if count not in (0, holders):
                raise RuntimeError(
                    f"Muon matrix {index}: {count} of its {holders} slice holders "
                    "have a gradient"
                )
            active.append(count == holders and holders > 0)
        return active

    def send_buffer(self, device) -> torch.Tensor:
        """This rank's bf16 slices of every matrix, grouped by owner rank."""

        return torch.empty(self.send_total, dtype=torch.bfloat16, device=device)

    def pack(self, send: torch.Tensor, index: int, update: torch.Tensor) -> None:
        start = self.send_pos[index]
        send[start : start + update.numel()].copy_(update.reshape(-1))

    def orthogonalize_local(self, updates, *, ns_steps, coefficients, eps):
        """Full matrices on one rank: index -> NewtonSchulz5(update) (bf16)."""

        return {
            index: zeropower_via_newtonschulz5(
                update.reshape(self.shapes[index]),
                ns_steps=ns_steps,
                coefficients=coefficients,
                eps=eps,
            )
            for index, update in updates.items()
        }

    def orthogonalize_sharded(
        self, send, active, local, *, ns_steps, coefficients, eps
    ):
        """Index -> this rank's slice of NewtonSchulz5(full update) (bf16).

        ``send`` holds this rank's packed bf16 slices (:meth:`pack`); ``local``
        lists the matrices with a non-empty local slice to return.
        """

        me = self.rank
        device = send.device
        recv = torch.empty(self.recv_total, dtype=torch.bfloat16, device=device)
        dist.all_to_all_single(
            recv, send, self.recv_splits, self.send_splits, group=self.group
        )
        reply = torch.empty_like(recv)
        for index in self.owned[me]:
            if not active[index]:
                continue
            full = torch.empty(
                math.prod(self.shapes[index]), dtype=torch.bfloat16, device=device
            )
            for source in range(self.world):
                size = self.sizes[source][index]
                if size:
                    start = self.recv_pos[(source, index)]
                    offset = self.offsets[source][index]
                    full[offset : offset + size].copy_(recv[start : start + size])
            ortho = zeropower_via_newtonschulz5(
                full.view(self.shapes[index]),
                ns_steps=ns_steps,
                coefficients=coefficients,
                eps=eps,
            ).reshape(-1)
            for source in range(self.world):
                size = self.sizes[source][index]
                if size:
                    start = self.recv_pos[(source, index)]
                    offset = self.offsets[source][index]
                    reply[start : start + size].copy_(ortho[offset : offset + size])
            del full, ortho
        del recv
        back = torch.empty_like(send)
        dist.all_to_all_single(
            back, reply, self.send_splits, self.recv_splits, group=self.group
        )
        return {
            index: back[
                self.send_pos[index] : self.send_pos[index] + self.sizes[me][index]
            ]
            for index in local
        }


class ShardedMuon(torch.optim.Optimizer):
    """Muon over fp32 masters that may be FSDP1 flat local slices.

    One parameter group. Matches ``torch.optim.Muon`` op for op on full
    matrices; on slices, see the module docstring. The state is ``torch.optim``
    standard (``momentum_buffer`` per rank-local slice).
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        *,
        names: Sequence[str],
        shapes: Sequence[Sequence[int]],
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adjust_lr_fn: str = "original",
    ) -> None:
        params = list(params)
        if not params:
            raise ValueError("Muon needs at least one matrix")
        if len(names) != len(params) or len(shapes) != len(params):
            raise ValueError("every Muon parameter needs a name and a logical shape")
        if not lr > 0:
            raise ValueError(f"Muon learning rate must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Muon momentum must be in [0, 1), got {momentum}")
        if not 0 < int(ns_steps) < 100:
            raise ValueError(f"Muon ns_steps must be in [1, 99], got {ns_steps}")
        if weight_decay < 0:
            raise ValueError(f"Muon weight decay must be >= 0, got {weight_decay}")
        if adjust_lr_fn not in MUON_LR_ADJUSTMENTS:
            raise ValueError(
                f"unsupported Muon adjust_lr_fn={adjust_lr_fn!r}; "
                f"expected one of {list(MUON_LR_ADJUSTMENTS)}"
            )
        for name, shape in zip(names, shapes):
            if len(shape) != 2:
                raise ValueError(
                    f"Muon parameter {name} is not a matrix: {tuple(shape)}"
                )
        defaults = dict(
            lr=lr,
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            weight_decay=float(weight_decay),
            ns_coefficients=tuple(NS_COEFFICIENTS),
            eps=NS_EPS,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        self.names = tuple(names)
        self.shapes = tuple(tuple(int(size) for size in shape) for shape in shapes)
        self._plan: Optional[_MuonPlan] = None
        if all(p.numel() == math.prod(shape) for p, shape in zip(params, self.shapes)):
            self._plan = _MuonPlan.local(self.shapes)

    @property
    def plan(self) -> Optional[_MuonPlan]:
        return self._plan

    def configure_sharding(self, *, process_group=None, enabled: bool = True) -> None:
        """Bind the group that owns disjoint slices (the grad-norm group)."""

        params = self.param_groups[0]["params"]
        sharded = (
            enabled
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(group=process_group) > 1
        )
        if not sharded:
            for name, parameter, shape in zip(self.names, params, self.shapes):
                if parameter.numel() != math.prod(shape):
                    raise RuntimeError(
                        f"Muon matrix {name} is a {parameter.numel()}-element local "
                        f"slice of {shape} but no sharded process group is configured"
                    )
            self._plan = _MuonPlan.local(self.shapes)
            return
        world = dist.get_world_size(group=process_group)
        local = torch.tensor(
            [p.numel() for p in params], dtype=torch.int64, device=params[0].device
        )
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local, group=process_group)
        sizes = [[int(size) for size in row.tolist()] for row in gathered]
        self._plan = _MuonPlan(
            self.shapes,
            sizes,
            assign_owners(self.shapes, world),
            rank=dist.get_rank(group=process_group),
            group=process_group,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("ShardedMuon.step does not take a closure")
        plan = self._plan
        if plan is None:
            raise RuntimeError(
                "Muon parameters are local slices but the sharded process group was "
                "never configured (BF16Optimizer.configure_grad_norm_reduction)"
            )
        if len(self.param_groups) != 1:
            raise RuntimeError("ShardedMuon has exactly one parameter group")
        group = self.param_groups[0]
        params = group["params"]
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        adjust_lr_fn = group["adjust_lr_fn"]
        active = plan.active(params)
        nesterov = group["nesterov"]
        ns = dict(
            ns_steps=group["ns_steps"],
            coefficients=group["ns_coefficients"],
            eps=group["eps"],
        )
        # Sharded: each slice's update goes straight into the bf16 send buffer.
        send = plan.send_buffer(params[0].device) if plan.sharded else None
        updates: Dict[int, torch.Tensor] = {}
        local: List[int] = []
        for index, parameter in enumerate(params):
            if not active[index] or not parameter.numel():
                continue
            grad = parameter.grad
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    grad, memory_format=torch.preserve_format
                )
            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1 - momentum)
            update = grad.lerp(buf, momentum) if nesterov else buf
            if send is None:
                updates[index] = update
            else:
                plan.pack(send, index, update)
                local.append(index)
            del update
        orthogonal = (
            plan.orthogonalize_local(updates, **ns)
            if send is None
            else plan.orthogonalize_sharded(send, active, local, **ns)
        )
        del updates, send
        for index, ortho in orthogonal.items():
            parameter = params[index]
            if weight_decay:
                parameter.mul_(1 - lr * weight_decay)
            parameter.add_(
                ortho.reshape(parameter.shape),
                alpha=-(lr * muon_lr_ratio(self.shapes[index], adjust_lr_fn)),
            )
        return None


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """The AdamW and Muon optimizers behind one ``Optimizer`` face.

    ``param_groups`` is ``[AdamW group, Muon group]`` (live views of the two
    optimizers' groups), so the one LR scheduler of ``BF16Optimizer`` applies
    one multiplier schedule to both peak rates, ``param_groups[0]`` stays the
    AdamW rate (the trainer's ``lr``), and the restore checks cover both groups.
    It owns no parameters or state; ``Optimizer.__init__`` is deliberately not run.
    """

    def __init__(self, adamw: torch.optim.Optimizer, muon: ShardedMuon) -> None:
        self.adamw = adamw
        self.muon = muon
        self.defaults = {}

    @property
    def param_groups(self):
        return [*self.adamw.param_groups, *self.muon.param_groups]

    @property
    def state(self):
        merged = dict(self.adamw.state)
        merged.update(self.muon.state)
        return merged

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MuonWithAuxAdamW.step does not take a closure")
        self.adamw.step()
        self.muon.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {ADAMW: self.adamw.state_dict(), MUON: self.muon.state_dict()}

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, Mapping) or set(state_dict) != {ADAMW, MUON}:
            keys = sorted(state_dict) if isinstance(state_dict, Mapping) else state_dict
            raise ValueError(
                f"Muon optimizer state must hold exactly {[ADAMW, MUON]}, got {keys}"
            )
        self.adamw.load_state_dict(state_dict[ADAMW])
        self.muon.load_state_dict(state_dict[MUON])

    def __repr__(self) -> str:
        return f"MuonWithAuxAdamW(adamw={self.adamw!r}, muon={self.muon!r})"


__all__ = [
    "ADAMW",
    "MUON",
    "MUON_GROUP_RULE",
    "MUON_LR_ADJUSTMENTS",
    "MuonWithAuxAdamW",
    "NS_COEFFICIENTS",
    "NS_EPS",
    "OPTIMIZER_KINDS",
    "ParameterInfo",
    "STATE_FORMAT",
    "ShardedMuon",
    "assign_owners",
    "canonical_parameter_name",
    "capture_parameter_layout",
    "muon_lr_ratio",
    "newton_schulz_cost",
    "zeropower_via_newtonschulz5",
]
