"""Shared fixtures of the Muon tests: a miniature DFlash-family draft, a pure
``torch.optim`` oracle (``torch.optim.Muon`` + ``torch.optim.AdamW`` on fp32
masters, with BF16Optimizer's global clip) and a bitwise tree comparison."""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

#: ``torch.optim.Muon`` (torch >= 2.9) is the oracle of the bitwise tests.
HAS_TORCH_MUON = hasattr(torch.optim, "Muon")

#: Draft naming at miniature size (hidden 8, 4 x 3 query heads, 2 kv heads,
#: MLP 20, GRU 5, emb 6, vocab 11, 2 layers).
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, MLP, GRU, EMB, VOCAB, LAYERS = (
    8,
    4,
    2,
    3,
    20,
    5,
    6,
    11,
    2,
)
MUON_NAME_RE = re.compile(
    r"^layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$"
)


class RMSNorm(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)
        return normed.to(x.dtype) * self.weight


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(self, h):
        batch = h.shape[0]
        q = self.q_norm(
            self.q_proj(h).view(batch, KV_HEADS, HEADS // KV_HEADS, HEAD_DIM)
        )
        k = self.k_norm(self.k_proj(h).view(batch, KV_HEADS, 1, HEAD_DIM))
        v = self.v_proj(h).view(batch, KV_HEADS, 1, HEAD_DIM)
        mixed = torch.sigmoid(q * k) * v
        return self.o_proj(mixed.reshape(batch, HEADS * HEAD_DIM))


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, MLP, bias=False)
        self.up_proj = nn.Linear(HIDDEN, MLP, bias=False)
        self.down_proj = nn.Linear(MLP, HIDDEN, bias=False)

    def forward(self, h):
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class TinyDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()
        self.input_layernorm = RMSNorm(HIDDEN)
        self.post_attention_layernorm = RMSNorm(HIDDEN)

    def forward(self, h):
        h = h + self.self_attn(self.input_layernorm(h))
        return h + self.mlp(self.post_attention_layernorm(h))


class TinyDraft(nn.Module):
    """Domino-style trainable names (fc, hidden_norm, norm, layers.*,
    prefix_gru, embed_proj.{0,2}) at miniature size, plus one frozen tensor."""

    _no_split_modules = ["TinyDecoderLayer"]

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TinyDecoderLayer() for _ in range(LAYERS))
        self.norm = RMSNorm(HIDDEN)
        self.fc = nn.Linear(3 * HIDDEN, HIDDEN, bias=False)
        self.hidden_norm = RMSNorm(HIDDEN)
        self.prefix_gru = nn.GRU(HIDDEN, GRU, batch_first=True, bias=False)
        self.embed_proj = nn.Sequential(
            nn.Linear(HIDDEN + GRU, EMB, bias=False),
            nn.SiLU(),
            nn.Linear(EMB, VOCAB, bias=False),
        )
        self.frozen_table = nn.Parameter(
            torch.randn(VOCAB, HIDDEN), requires_grad=False
        )

    def forward(self, x):
        h = self.hidden_norm(self.fc(x))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        gru, _ = self.prefix_gru(h.unsqueeze(1))
        logits = self.embed_proj(torch.cat([h, gru.squeeze(1)], dim=-1))
        target = (h @ self.frozen_table.T).float()
        return (logits.float() - target).square().mean()


class Composite(nn.Module):
    """The trainer's shape: the checkpointed, optimized unit is ``.draft_model``."""

    def __init__(self) -> None:
        super().__init__()
        self.draft_model = TinyDraft()

    def forward(self, x):
        return self.draft_model(x)


def tiny_inputs(step: int, batch: int = 3, dtype=torch.bfloat16, device=None):
    generator = torch.Generator().manual_seed(1000 + step)
    return torch.randn(batch, 3 * HIDDEN, generator=generator).to(
        device=device, dtype=dtype
    )


def tiny_draft(seed: int = 0, dtype=torch.bfloat16) -> TinyDraft:
    torch.manual_seed(seed)
    return TinyDraft().to(dtype)


def expected_muon_names(layers: int = LAYERS):
    return [
        f"layers.{i}.{name}.weight"
        for i in range(layers)
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]


def constant_warmup_lr(update: int, base_lr: float, warmup: int) -> float:
    """The ``constant`` schedule's rate for 1-based optimizer update ``update``.

    Same float expression as ``_WarmupScheduler.get_lr``
    (``last_epoch + 1 == update``), so the two agree bit for bit.
    """
    if update <= warmup:
        return update / warmup * base_lr
    return base_lr


class TorchReference:
    """``torch.optim.Muon`` on the Muon names and ``torch.optim.AdamW`` on the
    rest, over fp32 masters of ``model``'s trainable tensors, with
    BF16Optimizer's global clip and copy-back."""

    def __init__(
        self,
        model,
        *,
        lr,
        muon_lr,
        momentum=0.95,
        nesterov=True,
        muon_weight_decay=0.0,
        adjust_lr_fn="original",
        max_grad_norm=1.0,
        warmup=0,
        use_muon=True,
    ):
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.masters = [
            p.detach().clone().float().requires_grad_(True) for _, p in self.params
        ]
        on_muon = [use_muon and bool(MUON_NAME_RE.match(n)) for n, _ in self.params]
        muon = [m for m, flag in zip(self.masters, on_muon) if flag]
        adamw = [m for m, flag in zip(self.masters, on_muon) if not flag]
        self.muon = (
            torch.optim.Muon(
                muon,
                lr=muon_lr,
                weight_decay=muon_weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                adjust_lr_fn=adjust_lr_fn,
            )
            if muon
            else None
        )
        self.adamw = torch.optim.AdamW(adamw, lr=lr, weight_decay=0.0)
        self.lr, self.muon_lr = lr, muon_lr
        self.max_grad_norm = max_grad_norm
        self.warmup = warmup

    def _optimizers(self):
        return [self.adamw] + ([self.muon] if self.muon is not None else [])

    @torch.no_grad()
    def step(self, update: int):
        grads = [p.grad.detach() for _, p in self.params if p.grad is not None]
        total = torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
        coefficient = torch.clamp(self.max_grad_norm / (total + 1e-6), max=1.0)
        for master, (_, p) in zip(self.masters, self.params):
            if p.grad is None:
                master.grad = None
                continue
            grad = p.grad.detach().to(dtype=torch.float32)
            grad.mul_(coefficient)
            master.grad = grad
        self.adamw.param_groups[0]["lr"] = constant_warmup_lr(
            update, self.lr, self.warmup
        )
        if self.muon is not None:
            self.muon.param_groups[0]["lr"] = constant_warmup_lr(
                update, self.muon_lr, self.warmup
            )
        for optimizer in self._optimizers():
            optimizer.step()
            optimizer.zero_grad()
        for master, (_, p) in zip(self.masters, self.params):
            p.data.copy_(master.data.to(dtype=p.dtype))
            p.grad = None
        return total

    def master(self, name):
        return self.masters[[n for n, _ in self.params].index(name)]

    def momentum(self, name):
        return self.muon.state[self.master(name)]["momentum_buffer"]


def tree_differences(x, y, key="", diffs=None):
    """Bitwise differences between two checkpoint trees."""

    diffs = [] if diffs is None else diffs
    if torch.is_tensor(x):
        if not torch.is_tensor(y) or x.shape != y.shape or x.dtype != y.dtype:
            diffs.append((key, "tensor-meta"))
        elif not torch.equal(x, y):
            diffs.append((key, "tensor"))
    elif isinstance(x, dict):
        if not isinstance(y, dict) or set(x) != set(y):
            diffs.append((key, "dict-keys"))
            return diffs
        for name in x:
            tree_differences(x[name], y[name], f"{key}/{name}", diffs)
    elif isinstance(x, (list, tuple)):
        if not isinstance(y, (list, tuple)) or len(x) != len(y):
            diffs.append((key, "sequence-length"))
            return diffs
        for index, (a, b) in enumerate(zip(x, y)):
            tree_differences(a, b, f"{key}/{index}", diffs)
    elif (
        isinstance(x, float)
        and isinstance(y, float)
        and math.isnan(x)
        and math.isnan(y)
    ):
        pass
    elif x != y:
        diffs.append((key, f"value {x!r} != {y!r}"))
    return diffs
