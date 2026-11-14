"""DFlash2 unary CE plus covered-candidate selector CE."""

from dataclasses import dataclass

from specforge.data.loss_mask import has_consecutive_supervised_tokens

from ..base import check_options
from ..inputs import prepare_blocks
from ..losses import TokenLoss, decay_weighting
from ..plan import LossPlan, LossTerm
from ..predictions import PredictionBatch
from ..registry import TrainingRequirements, register_protocol
from .dflash import COMMON_MODULES


@register_protocol(
    "dflash2",
    legacy_strategy="dflash",
    training=TrainingRequirements(
        capture_method="dflash",
        minimum_loss_tokens=2,
        loss_mask_filter=has_consecutive_supervised_tokens,
    ),
    architectures=("DFlash2DraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    provides=(
        "logits",
        "labels",
        "teacher_logits",
        "chain_positions",
        "acceptance_chain",
    ),
    trains=COMMON_MODULES + ("candidate_selector",),
)
@dataclass
class DFlash2Objective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None
    selector_loss_alpha: float = 1.0
    selector_warmup_steps: int | None = None
    selector_ramp_steps: int | None = None
    selector_warmup_ratio: float = 0.0
    selector_ramp_ratio: float = 0.0
    selector_stop_gradient: bool = False

    default_loss = TokenLoss(kind="ce")
    default_weighting_options = ("loss_decay_gamma",)

    @property
    def default_weighting(self):
        return decay_weighting(self.loss_decay_gamma)

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={"selector_loss_alpha": self.selector_loss_alpha},
            probabilities={
                "selector_warmup_ratio": self.selector_warmup_ratio,
                "selector_ramp_ratio": self.selector_ramp_ratio,
            },
            zero_counts={
                "selector_warmup_steps": self.selector_warmup_steps,
                "selector_ramp_steps": self.selector_ramp_steps,
            },
        )
        for name in ("warmup", "ramp"):
            if getattr(self, f"selector_{name}_steps") is not None and getattr(
                self, f"selector_{name}_ratio"
            ):
                raise ValueError(f"selector {name} must use steps or ratio, not both")
        if type(self.selector_stop_gradient) is not bool:
            raise ValueError("selector_stop_gradient must be boolean")

    @property
    def frozen_parameters(self):
        return ("candidate_selector",) if self.selector_loss_alpha == 0 else ()

    def selector_scale(self, step, total_steps):
        if (self.selector_warmup_ratio or self.selector_ramp_ratio) and not total_steps:
            raise ValueError("selector ratio schedule requires total_steps")
        warmup = self.selector_warmup_steps
        ramp = self.selector_ramp_steps
        if warmup is None:
            warmup = int((total_steps or 0) * self.selector_warmup_ratio)
        if ramp is None:
            ramp = int((total_steps or 0) * self.selector_ramp_ratio)
        if step < warmup:
            return 0.0
        return min((step - warmup + 1) / ramp, 1.0) if ramp else 1.0

    def __call__(self, runtime, batch, *, step, total_steps, **context):
        model = runtime.draft_model
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            output_logits=False,
        )
        scale = self.selector_loss_alpha * self.selector_scale(step, total_steps)

        def compute(calc):
            calc.run((0,))
            predictions, labels = calc.predictions, calc.labels
            terms = calc.evaluate(
                "candidates",
                TokenLoss(
                    "ce", {"top_k": model.config.draft.spec_config.selector_top_k}
                ),
                calc.outputs()["logits"],
                labels,
            )
            ids = terms.statistics["topk_ids"]
            unary, hidden = terms.statistics["topk_values"], predictions.hidden
            if self.selector_stop_gradient:
                unary, hidden = unary.detach(), hidden.detach()
            scores = model.candidate_selector.score_candidates(
                candidate_ids=ids,
                unary_logits=unary,
                hidden_states=hidden,
                predecessor_ids=predictions.context_inputs["predecessors"],
            )
            covered = ids == labels[..., None]
            selector = calc.evaluate(
                "selector", TokenLoss("ce"), scores, covered.long().argmax(-1)
            ).values * covered.any(-1)
            calc.add(1, selector)

        predictions = runtime.block_predictions(
            output,
            blocks,
            batch,
            compute=compute,
            plan=LossPlan(
                (LossTerm(), LossTerm("selector", scale, TokenLoss("value")))
            ),
            context_inputs={"predecessors": blocks.predecessors},
        )
        return PredictionBatch((predictions,))
