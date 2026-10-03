"""SimpleFSDP model registration for the native TorchTitan GraphTrainer."""

from torch.fx.traceback import annotate_fn
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile


def parallelize_graph_dflash(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    """Use graph-native parameter collectives without eager FSDP hooks."""
    if parallel_dims.tp_enabled or parallel_dims.cp_enabled or parallel_dims.pp_enabled:
        raise ValueError(
            "SpecForge GraphTrainer currently supports data parallelism only"
        )
    if parallelism.spmd_backend != "partial_dtensor":
        raise ValueError("SpecForge GraphTrainer requires partial_dtensor")
    if ac_config is not None:
        raise ValueError("GraphTrainer manages activation memory through graph passes")
    # GraphTrainer's native block bucketing consumes ``layers.N`` annotations.
    # Expose the draft's natural module names rather than adapter prefixes.
    annotate_module_fqns(model.draft_model)
    model.training_model.forward = annotate_fn({"module_fqn": "loss"})(
        model.training_model.forward
    )
    # Frozen teacher weights remain replicated. SimpleFSDP's parameter wrapping
    # constructs trainable Parameters, so apply it only to the trainable draft.
    apply_simple_fsdp(model.draft_model, parallel_dims=parallel_dims, training=training)
    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
    model.training_model.objective_process_group = parallel_dims.get_optional_mesh(
        "loss", include_singleton_axes=True
    ).get_group()
    model.training_model.objective_context_group = parallel_dims.get_optional_mesh(
        "cp", include_singleton_axes=True
    ).get_group()
    return model
