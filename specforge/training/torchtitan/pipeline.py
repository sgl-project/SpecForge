"""DFlash pipeline stages carrying draft activations and projected context."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.pipelining import PipelineStage
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.pipeline_parallel import _build_pipeline_schedule

from specforge.algorithms.common.dflash_family_model import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.flex_attention_backend import flex_attention_backend

from .model import SpecForgeTitanModel


class PipelineDecoder(nn.Module):
    """A contiguous subset of decoder layers with unchanged state-dict names.

    The projected teacher context is an activation, not a detached stage-local
    copy. Its backward contributions from all later layers flow back to the
    first stage's fc/hidden_norm parameters through the pipeline schedule.
    """

    def __init__(self, draft, *, first: bool, last: bool, start: int, stop: int):
        super().__init__()
        for name, value in vars(draft).items():
            if not name.startswith("_"):
                setattr(self, name, value)
        for name, module in draft.named_children():
            if name == "layers":
                module = nn.ModuleDict(
                    {str(i): draft.layers[i] for i in range(start, stop)}
                )
            elif name in {"fc", "hidden_norm"} and not first:
                module = None
            elif (
                name in {"norm", "candidate_selector", "markov_head", "confidence_head"}
                and not last
            ):
                module = None
            setattr(self, name, module)
        self.first, self.last = first, last
        self.enable_weight_tying = False
        self.tok_embeddings = None
        self.lm_head = None
        for name in ("norm", "candidate_selector", "markov_head", "confidence_head"):
            if not hasattr(self, name):
                setattr(self, name, None)

    # Reuse the algorithm's public output-head operations verbatim. The stage
    # owns their original modules only on the final rank.
    _dflash2_config = DFlash2DraftModel._dflash2_config

    def unary_logits_transform_is_identity(self):
        return (
            self.candidate_selector is not None
            and DFlash2DraftModel.unary_logits_transform_is_identity(self)
        )

    transform_unary_logits = DFlash2DraftModel.transform_unary_logits
    apply_logits_head = DSparkDraftModel.apply_logits_head
    predict_confidence = DSparkDraftModel.predict_confidence

    def forward(self, hidden, context, *, position_ids, attention_mask, **kwargs):
        if self.first:
            context = self.hidden_norm(self.fc(context))
        position_embeddings = self.rotary_emb(hidden, position_ids)
        for name, layer in self.layers.items():
            mask = (
                attention_mask[self.layer_types[int(name)]]
                if isinstance(attention_mask, dict)
                else attention_mask
            )
            hidden = layer(
                hidden_states=hidden,
                target_hidden=context,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
                **kwargs,
            )
        if self.last:
            hidden = self.norm(hidden)
        return hidden, context


class SpecForgePipelineStage(SpecForgeTitanModel):
    def __init__(self, model, *, stage_index: int, num_stages: int):
        nn.Module.__init__(self)
        self.config = model.config
        self.training_model = copy.deepcopy(model.training_model)
        self.first = stage_index == 0
        self.last = stage_index == num_stages - 1
        self._is_pipeline_stage = True
        layers = len(self.draft_model.layers)
        self.training_model.draft_model = PipelineDecoder(
            self.draft_model,
            first=self.first,
            last=self.last,
            start=layers * stage_index // num_stages,
            stop=layers * (stage_index + 1) // num_stages,
        )

    def forward(self, inputs, context=None, *, source_input_ids, **kwargs):
        objective = self.training_model
        anchors, keep = kwargs["anchor_positions"], kwargs["block_keep_mask"]
        seq_len = source_input_ids.shape[1]
        positions = torch.cat(
            [
                torch.arange(seq_len, device=inputs.device)
                .unsqueeze(0)
                .expand(source_input_ids.shape[0], -1),
                objective._create_position_ids(anchors),
            ],
            dim=1,
        )
        mask_builder = (
            create_dflash_block_mask
            if objective.attention_backend == "flex_attention"
            else create_dflash_sdpa_mask
        )
        mask_args = dict(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=seq_len,
            block_size=objective.block_size,
            device=inputs.device,
        )
        kernel_kwargs = {}
        if objective.attention_backend == "flex_attention":
            if flex_attention_backend() == "FLASH":
                mask_args["flex_block_size"] = (256, 128)
            kernel_kwargs["kernel_options"] = {"BACKEND": "TRITON"}
        if self.draft_model.sliding_window is None:
            attention_mask = mask_builder(**mask_args)
        else:
            attention_mask = {
                kind: mask_builder(
                    **mask_args,
                    **(
                        {"sliding_window": self.draft_model.sliding_window}
                        if kind == "sliding_attention"
                        else {}
                    ),
                )
                for kind in set(self.draft_model.layer_types)
            }
        if self.first:
            inputs = objective._create_noise_embed(source_input_ids, anchors, keep)
            context = kwargs["hidden_states"]
        hidden, context = self.draft_model(
            inputs,
            context,
            position_ids=positions,
            attention_mask=attention_mask,
            **kernel_kwargs,
        )
        if not self.last:
            return hidden, context
        normalizer = kwargs.pop("objective_normalizer")
        loss, _accuracy, metrics = objective(
            input_ids=source_input_ids,
            output_hidden=hidden,
            **kwargs,
        )
        numerator = metrics["loss_terms"][0] if "loss_terms" in metrics else loss
        # PipelineStage exchanges tensor metadata; nested metric dictionaries
        # do not belong on a stage boundary. The native schedule owns backward.
        return torch.stack((numerator, normalizer))


def pipeline_dflash(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
    device,
    model_config,
    parallelize_fn,
    loss_fn,
):
    del model_config
    if parallelism.pipeline_parallel_schedule not in {"1F1B", "GPipe"}:
        raise ValueError("DFlash currently supports 1F1B or GPipe pipeline schedules")
    mesh = parallel_dims.get_mesh("pp")
    index, degree = mesh.get_local_rank(), mesh.size()
    if len(model.draft_model.layers) < degree:
        raise ValueError("DFlash pipeline degree cannot exceed decoder layer count")
    stage_model = SpecForgePipelineStage(model, stage_index=index, num_stages=degree)
    stage_model = parallelize_fn(
        stage_model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
    # PipelineStage coordinates FSDP accumulation only when its top-level
    # submodule is an FSDPModule. The objective reads selector/Markov weights
    # after decoder.forward(), so their storage must survive the earlier
    # microbatch backwards. This empty outer FSDP group lets the native
    # schedule set recursive sync/reshard flags; teacher tables stay replicated.
    dp_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    fully_shard(
        stage_model,
        mesh=parallel_dims.get_mesh(dp_names),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cast_forward_inputs=False,
        ),
        reshard_after_forward=False,
        ignored_params={
            parameter
            for parameter in stage_model.parameters()
            if not parameter.requires_grad
        },
    )
    stage = PipelineStage(stage_model, index, degree, device, group=mesh.get_group())
    schedule = _build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=[stage],
        loss_fn=loss_fn,
    )
    return schedule, [stage_model], index == 0, index == degree - 1
