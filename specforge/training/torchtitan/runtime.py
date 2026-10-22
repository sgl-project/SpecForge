"""Native TorchTitan trainer with SpecForge feature preparation hooks.

The constructor, training loop, accumulation, backward, clipping, optimizer,
scheduler and distributed checkpoints are TorchTitan's implementations. Only
feature preparation and the additional RNG checkpoint state live here.
"""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torchtitan.trainer import Trainer

from .metrics import ObjectiveMetricLogger
from .numerics import compiler_numerics
from .parallelize import HeterogeneousGradientNorms


class SpecForgeTitanTrainer(Trainer):
    _fixed_graph_shapes = False

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        resume_contract: dict = field(default_factory=dict)
        schedule_total_steps: int | None = None

    def __init__(self, config: Config):
        self._fixed_graph_shapes = (
            not config.training.disable_cuda_graphs
            or getattr(getattr(config, "compile", None), "mode", None) == "aot_fx_trace"
        )
        if self._fixed_graph_shapes and config.parallelism.pipeline_parallel_degree > 1:
            raise ValueError(
                "SpecForge graph training currently requires pipeline_parallel_degree=1"
            )
        if (
            config.parallelism.pipeline_parallel_degree > 1
            and config.model_spec is not None
            and config.model_spec.model.objective.get("lk_loss_type") == "lambda"
        ):
            raise ValueError(
                "Pipeline parallelism does not support LK-lambda: its nonlinear "
                "local-batch coefficient cannot be computed independently per "
                "pipeline microbatch"
            )
        self._anchor_generator = torch.Generator(device="cpu")
        super().__init__(config)
        self.metrics_processor.logger = ObjectiveMetricLogger(
            self.metrics_processor.logger
        )
        batch_mesh = self.parallel_dims.get_optional_mesh(
            "batch", include_singleton_axes=True
        )
        seed = config.debug.seed if config.debug.seed is not None else 42
        self._anchor_generator.manual_seed(seed + batch_mesh.get_local_rank())

    def batch_generator(self, data_iterable):
        """Prepare one optimizer window without retaining forward graphs.

        DFlash's denominator depends on the sampled blocks, not the number of
        labels. Sample once on CPU, share those anchors with model forward, and
        sum the exact objective measure across microbatches and DP ranks before
        TorchTitan starts its usual backward/optimizer sequence.
        """
        batches = super().batch_generator(data_iterable)
        model = self.model_parts[0].training_model
        algorithm = self.config.model_spec.model.algorithm
        batch_mesh = self.parallel_dims.get_optional_mesh(
            "batch", include_singleton_axes=True
        )
        context_mesh = self.parallel_dims.get_optional_mesh(
            "cp", include_singleton_axes=True
        )
        while True:
            window = []
            denominator = torch.zeros(
                (self.gradient_accumulation_steps,) if algorithm == "dspark" else (),
                dtype=torch.float32,
            )
            for _ in range(
                self.gradient_accumulation_steps
                * self.num_pipeline_parallel_microbatches
            ):
                inputs, labels = next(batches)
                mask = inputs["loss_mask"]
                anchors, keep = model._sample_anchor_positions(
                    mask.shape[1],
                    mask,
                    torch.device("cpu"),
                    generator=self._anchor_generator,
                )
                inputs = dict(inputs)
                inputs["collect_detailed_metrics"] = (
                    False
                    if self._fixed_graph_shapes
                    else self.metrics_processor.should_log(self.step)
                )
                local_den = model.prepared_objective_denominator(mask, anchors, keep)
                if algorithm == "dspark":
                    # DSpark averages logical local-batch objectives over the
                    # accumulation window. PP subdivisions share one logical
                    # batch denominator even when valid counts differ.
                    group_index = len(window) // self.num_pipeline_parallel_microbatches
                    denominator[group_index] += local_den
                else:
                    denominator += local_den
                    alpha = self._selector_alpha(model)
                    inputs["selector_loss_alpha"] = (
                        torch.tensor(alpha, dtype=torch.float32)
                        if self._fixed_graph_shapes
                        else alpha
                    )
                    if model.loss_type != "dflash":
                        _, weights = model._dflash_weight_mask(mask, anchors, keep)
                        scale = model._sequence_anchor_scale(weights).squeeze(-1)
                        local_scale, _ = partition_anchor_blocks(
                            scale,
                            keep,
                            rank=context_mesh.get_local_rank(),
                            degree=context_mesh.size(),
                            capacity=(
                                model.num_anchors
                                if self.parallel_dims.pp_enabled
                                or self._fixed_graph_shapes
                                else None
                            ),
                        )
                        inputs["prepared_sequence_anchor_scale"] = (
                            local_scale.unsqueeze(-1)
                        )
                anchors, keep = partition_anchor_blocks(
                    anchors,
                    keep,
                    rank=context_mesh.get_local_rank(),
                    degree=context_mesh.size(),
                    capacity=(
                        model.num_anchors
                        if self.parallel_dims.pp_enabled or self._fixed_graph_shapes
                        else None
                    ),
                )
                inputs["anchor_positions"] = anchors
                inputs["block_keep_mask"] = keep
                if context_mesh.size() > 1:
                    full_tokens = labels.numel()
                    labels = labels[
                        :, context_mesh.get_local_rank() :: context_mesh.size()
                    ]
                    self.metrics_processor.ntokens_since_last_log -= (
                        full_tokens - labels.numel()
                    )
                window.append((inputs, labels))
            normalizer = denominator.to(self.device)
            if batch_mesh.size() > 1:
                dist.all_reduce(
                    normalizer, op=dist.ReduceOp.SUM, group=batch_mesh.get_group()
                )
            torch._assert_async(
                (normalizer > 0).all(), "DFlash objective has no supervised targets"
            )
            for index, (inputs, labels) in enumerate(window):
                inputs["objective_normalizer"] = (
                    normalizer[index // self.num_pipeline_parallel_microbatches]
                    * self.gradient_accumulation_steps
                    if algorithm == "dspark"
                    else normalizer
                )
                yield inputs, labels

    def _selector_alpha(self, model):
        target = model.selector_loss_alpha
        total_steps = self.config.schedule_total_steps or self.config.training.steps
        # Titan increments self.step before fetching the window; SpecForge's
        # schedule receives the number of already-completed optimizer steps.
        completed = self.step - 1
        warmup = int(total_steps * model.selector_warmup_ratio)
        if completed < warmup:
            return 0.0
        ramp = int(total_steps * model.selector_ramp_ratio)
        if ramp <= 0:
            return target
        return target * min(max((completed - warmup + 1) / ramp, 0.0), 1.0)

    def train_step(self, data_iterator):
        # External model heads use a DP-only mesh while TP blocks use DP×TP.
        # Resolve only heterogeneous scalar gradient norms in a thread-local
        # dispatch scope; Titan still owns clipping, accumulation and stepping.
        with compiler_numerics():
            if self.parallel_dims.tp_enabled:
                with HeterogeneousGradientNorms():
                    return super().train_step(data_iterator)
            return super().train_step(data_iterator)

    def post_dataloading_process(self, input_dict, labels):
        # partial_dtensor TP operates inside the registered model's parallelize
        # plan. These feature tensors are replicated over TP and partitioned
        # over DP by the loader; generic causal-LM input sharding is incorrect.
        self.ntokens_seen += labels.numel()
        extra = {name: value for name, value in input_dict.items() if name != "input"}
        if self.parallel_dims.pp_enabled:
            extra["source_input_ids"] = input_dict["input"]
        return input_dict["input"], labels, extra

    def state_dict(self):
        state = super().state_dict()
        state[f"specforge_rng_rank_{dist.get_rank()}"] = {
            "anchor": self._anchor_generator.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.device),
        }
        state["specforge_world_size"] = dist.get_world_size()
        state["specforge_resume_contract"] = self.config.resume_contract
        return state

    def load_state_dict(self, state_dict):
        if state_dict["specforge_world_size"] != dist.get_world_size():
            raise ValueError("SpecForge RNG resume requires unchanged world size")
        if state_dict["specforge_resume_contract"] != self.config.resume_contract:
            raise ValueError(
                "SpecForge TorchTitan checkpoint training contract changed"
            )
        super().load_state_dict(state_dict)
        rng = state_dict[f"specforge_rng_rank_{dist.get_rank()}"]
        self._anchor_generator.set_state(rng["anchor"].cpu())
        torch.set_rng_state(rng["cpu"].cpu())
        torch.cuda.set_rng_state(rng["cuda"].cpu(), self.device)


def partition_anchor_blocks(anchors, keep, *, rank, degree, capacity=None):
    """Partition independent draft query blocks; teacher K/V stays replicated.

    Padding keeps every CP rank (and every PP microbatch) at the same width,
    including ranks with no valid anchors. The keep mask excludes padding from
    attention and objective weights. No block is split between CP ranks.
    """
    width = anchors.shape[1] if capacity is None else capacity
    width = ((width + degree - 1) // degree) * degree
    if width < anchors.shape[1]:
        raise ValueError("Anchor capacity is smaller than the prepared batch")
    if width > anchors.shape[1]:
        padding = width - anchors.shape[1]
        anchors = torch.nn.functional.pad(anchors, (0, padding))
        keep = torch.nn.functional.pad(keep, (0, padding), value=False)
    local_width = width // degree
    start = rank * local_width
    return anchors[:, start : start + local_width], keep[:, start : start + local_width]
