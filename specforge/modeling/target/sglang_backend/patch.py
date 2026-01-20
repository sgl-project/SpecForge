import logging
from typing import Optional

import sglang.srt.distributed.parallel_state as parallel_state
import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import init_model_parallel_group
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    compute_dp_attention_local_info,
    compute_dp_attention_world_info,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var

from specforge.distributed import get_tp_group as get_specforge_tp_group

logger = logging.getLogger(__name__)


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.debug(
        "world_size=%d rank=%d backend=%s",
        world_size,
        rank,
        backend,
    )
    assert (
        torch.distributed.is_initialized()
    ), "distributed environment should be initialized first"

    tp_group = get_specforge_tp_group()
    world_size = dist.get_world_size()
    tp_size = dist.get_world_size(tp_group)
    num_tp_groups = world_size // tp_size
    tp_ranks = []
    for i in range(num_tp_groups):
        tp_ranks.append(list(range(i * tp_size, (i + 1) * tp_size)))

    parallel_state._WORLD = GroupCoordinator(
        group_ranks=tp_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="world",
    )
    # we destroy the newly created world group and replace it
    # with the existing tp group from specforge to save CUDA memory
    group_to_destroy = parallel_state._WORLD.device_group
    parallel_state._WORLD.device_group = tp_group
    dist.destroy_process_group(group_to_destroy)


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    torch_compile: Optional[bool] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = parallel_state._WORLD.world_size
    backend = backend or dist.get_backend(parallel_state._WORLD.device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = (
        dist.get_world_size() // tensor_model_parallel_size
    )
    assert (
        parallel_state._TP is None
    ), "tensor model parallel group is already initialized"
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        )
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    parallel_state._TP = init_model_parallel_group(
        group_ranks,
        parallel_state._WORLD.local_rank,
        backend,
        use_message_queue_broadcaster=get_bool_env_var(
            "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "true"
        ),
        group_name="tp",
        pynccl_use_current_stream=duplicate_tp_group,
        torch_compile=torch_compile,
    )

    if duplicate_tp_group:
        assert (
            parallel_state._PDMUX_PREFILL_TP_GROUP is None
        ), "tensor model parallel group for PD-Multiplexing Prefill is already initialized"
        assert (
            parallel_state._PDMUX_PREFILL_TP_GROUP is None
        ), "tensor model parallel group for PD-Multiplexing Prefill is already initialized"
        parallel_state._PDMUX_PREFILL_TP_GROUP = init_model_parallel_group(
            group_ranks,
            parallel_state._WORLD.local_rank,
            backend,
            use_message_queue_broadcaster=get_bool_env_var(
                "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "true"
            ),
            group_name="pdmux_prefill_tp",
            pynccl_use_current_stream=True,
            torch_compile=torch_compile,
        )
        parallel_state._TP.pynccl_comm.disabled = False
        parallel_state._PDMUX_PREFILL_TP_GROUP.pynccl_comm.disabled = False

    moe_ep_size = expert_model_parallel_size

    moe_tp_size = tensor_model_parallel_size // moe_ep_size
    assert (
        parallel_state._MOE_EP is None
    ), "expert model parallel group is already initialized"
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        for j in range(moe_tp_size):
            st = i * tensor_model_parallel_size + j
            en = (i + 1) * tensor_model_parallel_size + j
            ranks = list(range(st, en, moe_tp_size))
            group_ranks.append(ranks)

    parallel_state._MOE_EP = init_model_parallel_group(
        group_ranks,
        parallel_state._WORLD.local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="moe_ep",
    )

    assert (
        parallel_state._MOE_TP is None
    ), "moe tensor model parallel group is already initialized"
    if moe_ep_size == 1:
        parallel_state._MOE_TP = parallel_state._TP
    else:
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            for j in range(moe_ep_size):
                st = i * tensor_model_parallel_size + j * moe_tp_size
                en = i * tensor_model_parallel_size + (j + 1) * moe_tp_size
                ranks = list(range(st, en))
                group_ranks.append(ranks)
        parallel_state._MOE_TP = init_model_parallel_group(
            group_ranks,
            parallel_state._WORLD.local_rank,
            backend,
            use_custom_allreduce=False,
            group_name="moe_tp",
        )

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = (
        dist.get_world_size() // pipeline_model_parallel_size
    )
    assert (
        parallel_state._PP is None
    ), "pipeline model parallel group is already initialized"
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(
            range(i, dist.get_world_size(), num_pipeline_model_parallel_groups)
        )
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    parallel_state._PP = init_model_parallel_group(
        group_ranks,
        parallel_state._WORLD.local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="pp",
    )


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    import sglang.srt.layers.dp_attention as dp_attention
    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    enable_dp_attention = server_args.enable_dp_attention
    tp_size = server_args.tp_size
    dp_size = server_args.dp_size
    moe_dense_tp_size = server_args.moe_dense_tp_size
    pp_size = server_args.pp_size

    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    dp_attention._ENABLE_DP_ATTENTION_FLAG = enable_dp_attention

    (
        dp_attention._ATTN_TP_RANK,
        dp_attention._ATTN_TP_SIZE,
        dp_attention._ATTN_DP_RANK,
    ) = compute_dp_attention_world_info(enable_dp_attention, tp_rank, tp_size, dp_size)
    _, _, dp_attention._LOCAL_ATTN_DP_RANK = compute_dp_attention_local_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
    )

    if enable_dp_attention:
        dp_attention._ATTN_DP_SIZE = dp_size
        if moe_dense_tp_size is None:
            dp_attention._LOCAL_ATTN_DP_SIZE = dp_attention._ATTN_DP_SIZE
        else:
            dp_attention._LOCAL_ATTN_DP_SIZE = max(
                1, dp_size // (tp_size // moe_dense_tp_size)
            )
    else:
        dp_attention._ATTN_DP_SIZE = 1
        dp_attention._LOCAL_ATTN_DP_SIZE = 1

    tp_group = parallel_state.get_tp_group()
    num_model_parallel_groups = dist.get_world_size() // (pp_size * tp_size)
    mp_size = pp_size * tp_size
    group_ranks = []

    for i in range(num_model_parallel_groups):
        ranks = [
            list(range(head, head + dp_attention._ATTN_TP_SIZE))
            for head in range(
                mp_size * i, mp_size * (i + 1), dp_attention._ATTN_TP_SIZE
            )
        ]
        group_ranks.extend(ranks)

    dp_attention._ATTN_TP_GROUP = GroupCoordinator(
        group_ranks,
        tp_group.local_rank,
        torch.distributed.get_backend(tp_group.device_group),
        use_pynccl=SYNC_TOKEN_IDS_ACROSS_TP,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="attention_tp",
    )
    # print(f"{parallel_state._ATTN_TP_GROUP=}")

    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )
