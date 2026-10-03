"""Experimental SpecForge adapter for TorchTitan's actual GraphTrainer."""

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from torch._guards import detect_fake_mode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta, set_current_meta
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.passes import construct_default_graph_passes
from torchtitan.experiments.graph_trainer.registry import register_pass_pipeline
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from .model import SpecForgeTitanModel
from .parameter_cache import graph_parameter_cache
from .runtime import SpecForgeTitanTrainer


class _MetadataInterpreter(torch.fx.Interpreter):
    def run_node(self, node):
        with set_current_meta(node, "specforge_functionalize"):
            return super().run_node(node)


def functionalize_fused_kernels(gm, example_inputs):
    """Make Triton output writes explicit before graph DCE and rematerialization.

    ``wrap_triton`` traces buffer writes as mutation-only nodes. The native
    GraphTrainer pass pipeline assumes functional dependencies; otherwise DCE
    drops kernels and leaves their output buffers uninitialized. Preserve block
    and backward metadata while applying PyTorch's own alias-aware transform.
    """
    if not any(
        "triton_kernel_wrapper_mutation" in str(n.target) for n in gm.graph.nodes
    ):
        return gm
    fake_mode = detect_fake_mode(example_inputs)
    with fake_mode if fake_mode is not None else nullcontext(), preserve_node_meta():
        result = make_fx(torch.func.functionalize(_MetadataInterpreter(gm).run))(
            *example_inputs
        )
    result.meta.update(gm.meta)
    return result


@register_pass_pipeline("specforge")
def specforge_graph_passes(traced_result, config, *, parallel_dims=None):
    return [functionalize_fused_kernels] + construct_default_graph_passes(
        traced_result, config, parallel_dims=parallel_dims
    )


class SpecForgeGraphModel(SpecForgeTitanModel):
    def forward(self, *args, **kwargs):
        with graph_parameter_cache(self):
            return super().forward(*args, **kwargs)

    @dataclass(kw_only=True, slots=True)
    class Config(SpecForgeTitanModel.Config):
        @property
        def layers(self):
            # Native bucketing needs the decoder count and optional MoE config.
            # All DFlash-family decoder blocks are dense.
            return (None,) * self.draft_config["num_hidden_layers"]


class SpecForgeGraphTrainer(SpecForgeTitanTrainer, GraphTrainer):
    """Reuse SpecForge preparation and native joint forward/loss/backward tracing.

    The cooperative MRO keeps the native GraphTrainer constructor, graph passes,
    gradient accumulation and teardown, alongside SpecForge's data/RNG hooks.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SpecForgeTitanTrainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config):
        if not config.training.disable_cuda_graphs:
            raise ValueError(
                "GraphTrainer requires the eager CUDA graph wrapper disabled; "
                "control graph capture through compile.disable_passes instead"
            )
        config.compile.pass_pipeline = "specforge"
        super().__init__(config)

    def forward_backward_step(self, **kwargs):
        # Native GraphTrainer assigns explicit graph gradient outputs directly
        # to empty param.grad slots. CUDA replay owns those output buffers and
        # overwrites them on the next microbatch. Give the accumulated gradient
        # its own storage before that next replay, preserving native tracing,
        # accumulation and optimizer logic. Later microbatches already own it.
        unowned = (
            [
                parameter
                for model in self.model_parts
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if self.gradient_accumulation_steps > 1
            else ()
        )
        # Keep repeated parameter reads shared through backward recomputation,
        # matching eager FSDP2's single unsharded BF16 leaf per parameter.
        with graph_parameter_cache(self.model_parts[0]):
            loss = super().forward_backward_step(**kwargs)
        for parameter in unowned:
            if parameter.grad is not None:
                parameter.grad = parameter.grad.clone()
        return loss
