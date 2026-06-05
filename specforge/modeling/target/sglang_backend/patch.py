import logging
from datetime import timedelta
from typing import Optional

import sglang.srt.distributed.parallel_state as parallel_state
import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import init_model_parallel_group
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    compute_dp_attention_local_info,
    compute_dp_attention_world_info,
)
from sglang.srt.server_args import ServerArgs

from specforge.distributed import get_tp_group as get_specforge_tp_group

logger = logging.getLogger(__name__)


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: Optional[int] = None,
    moe_a2a_backend: Optional[str] = None,
    recovered_rank: bool = False,
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    assert (
        torch.distributed.is_initialized()
    ), "distributed environment should be initialized first"

    if timeout is not None:
        assert isinstance(timeout, int), "timeout must be a number"
        assert timeout > 0, "timeout must be positive"
        parallel_state._MODEL_PARALLEL_GROUP_TIMEOUT = timedelta(seconds=timeout)

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
        recovered_rank=recovered_rank,
    )
    # Destroy the newly-created device group and replace it with the existing
    # SpecForge TP group to avoid allocating a duplicate NCCL communicator.
    group_to_destroy = parallel_state._WORLD.device_group
    parallel_state._WORLD.device_group = tp_group
    if group_to_destroy is not tp_group:
        try:
            dist.destroy_process_group(group_to_destroy)
        except Exception:
            logger.debug(
                "Failed to destroy temporary SGLang world group", exc_info=True
            )


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    attention_data_parallel_size: int = 1,
    attention_context_model_parallel_size: int = 1,
    moe_data_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    enable_symm_mem: bool = False,
    recovered_rank: bool = False,
) -> None:
    """Initialize SGLang model-parallel groups on top of SpecForge TP groups."""
    assert torch.distributed.is_initialized()
    world_size: int = parallel_state._WORLD.world_size
    distributed_world_size: int = dist.get_world_size()
    backend = backend or dist.get_backend(parallel_state._WORLD.device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    num_tensor_model_parallel_groups = (
        distributed_world_size // tensor_model_parallel_size
    )
    assert (
        parallel_state._TP is None
    ), "tensor model parallel group is already initialized"
    group_ranks = []
    for tp_group_idx in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(
                tp_group_idx * tensor_model_parallel_size,
                (tp_group_idx + 1) * tensor_model_parallel_size,
            )
        )
        group_ranks.append(ranks)

    parallel_state._TP = init_model_parallel_group(
        group_ranks,
        parallel_state._WORLD.local_rank,
        backend,
        use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
        group_name="tp",
        recovered_rank=recovered_rank,
    )

    if duplicate_tp_group:
        assert (
            parallel_state._PDMUX_PREFILL_TP_GROUP is None
        ), "tensor model parallel group for PD-Multiplexing Prefill is already initialized"
        parallel_state._PDMUX_PREFILL_TP_GROUP = init_model_parallel_group(
            group_ranks,
            parallel_state._WORLD.local_rank,
            backend,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="pdmux_prefill_tp",
            recovered_rank=recovered_rank,
        )
        if parallel_state._TP.pynccl_comm:
            parallel_state._TP.pynccl_comm.disabled = False
            parallel_state._PDMUX_PREFILL_TP_GROUP.pynccl_comm.disabled = False

    attn_dp_size = attention_data_parallel_size
    attn_cp_size = attention_context_model_parallel_size
    attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

    assert (
        parallel_state._ATTN_CP is None
    ), "attention context model parallel group is already initialized"
    if attn_cp_size == tensor_model_parallel_size:
        parallel_state._ATTN_CP = parallel_state._TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for dp_idx in range(attn_dp_size):
                for attn_tp_idx in range(attn_tp_size):
                    st = (
                        tp_group_idx * tensor_model_parallel_size
                        + dp_idx * attn_tp_size * attn_cp_size
                        + attn_tp_idx
                    )
                    en = (
                        tp_group_idx * tensor_model_parallel_size
                        + (dp_idx + 1) * attn_tp_size * attn_cp_size
                        + attn_tp_idx
                    )
                    ranks = list(range(st, en, attn_tp_size))
                    group_ranks.append(ranks)
        parallel_state._ATTN_CP = init_model_parallel_group(
            group_ranks,
            parallel_state._WORLD.local_rank,
            backend,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="attn_cp",
            recovered_rank=recovered_rank,
        )

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    assert (
        parallel_state._ATTN_TP is None
    ), "attention tensor model parallel group is already initialized"
    if attn_tp_size == tensor_model_parallel_size:
        parallel_state._ATTN_TP = parallel_state._TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for cp_dp_combined_idx in range(attn_cp_size * attn_dp_size):
                st = (
                    tp_group_idx * tensor_model_parallel_size
                    + cp_dp_combined_idx * attn_tp_size
                )
                en = (
                    tp_group_idx * tensor_model_parallel_size
                    + (cp_dp_combined_idx + 1) * attn_tp_size
                )
                ranks = list(range(st, en))
                group_ranks.append(ranks)

        parallel_state._ATTN_TP = init_model_parallel_group(
            group_ranks,
            parallel_state._WORLD.local_rank,
            backend,
            use_pynccl=SYNC_TOKEN_IDS_ACROSS_TP or enable_symm_mem,
            use_mscclpp_allreduce=False,
            use_custom_allreduce=False,
            use_torch_symm_mem_allreduce=False,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="attention_tp",
            recovered_rank=recovered_rank,
        )

    moe_ep_size = expert_model_parallel_size
    moe_dp_size = moe_data_model_parallel_size
    moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size

    assert (
        parallel_state._MOE_DP is None
    ), "moe data parallel group is already initialized"
    if attn_cp_size > moe_dp_size:
        parallel_state._MOE_DP = parallel_state._ATTN_CP
    elif moe_dp_size == tensor_model_parallel_size:
        parallel_state._MOE_DP = parallel_state._TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for tp_ep_combined_idx in range(moe_tp_size * moe_ep_size):
                st = tp_group_idx * tensor_model_parallel_size + tp_ep_combined_idx
                en = (
                    tp_group_idx + 1
                ) * tensor_model_parallel_size + tp_ep_combined_idx
                ranks = list(range(st, en, moe_tp_size * moe_ep_size))
                group_ranks.append(ranks)
        parallel_state._MOE_DP = init_model_parallel_group(
            group_ranks,
            parallel_state._WORLD.local_rank,
            backend,
            group_name="moe_dp",
            recovered_rank=recovered_rank,
        )

    assert (
        parallel_state._MOE_EP is None
    ), "expert model parallel group is already initialized"
    if moe_ep_size == tensor_model_parallel_size:
        parallel_state._MOE_EP = parallel_state._TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for moe_dp_idx in range(moe_dp_size):
                for moe_tp_idx in range(moe_tp_size):
                    st = (
                        tp_group_idx * tensor_model_parallel_size
                        + moe_dp_idx * moe_ep_size * moe_tp_size
                        + moe_tp_idx
                    )
                    en = st + moe_ep_size * moe_tp_size
                    ranks = list(range(st, en, moe_tp_size))
                    group_ranks.append(ranks)
        parallel_state._MOE_EP = init_model_parallel_group(
            group_ranks,
            parallel_state._WORLD.local_rank,
            backend,
            use_pynccl=False,
            use_custom_allreduce=False,
            group_name="moe_ep",
            recovered_rank=recovered_rank,
        )

    assert (
        parallel_state._MOE_TP is None
    ), "expert model parallel group is already initialized"
    if moe_tp_size == tensor_model_parallel_size:
        parallel_state._MOE_TP = parallel_state._TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for ep_dp_combined_idx in range(moe_ep_size * moe_dp_size):
                st = (
                    tp_group_idx * tensor_model_parallel_size
                    + ep_dp_combined_idx * moe_tp_size
                )
                en = (
                    tp_group_idx * tensor_model_parallel_size
                    + (ep_dp_combined_idx + 1) * moe_tp_size
                )
                ranks = list(range(st, en))
                group_ranks.append(ranks)
        parallel_state._MOE_TP = init_model_parallel_group(
            group_ranks,
            parallel_state._WORLD.local_rank,
            backend,
            use_pynccl=False,
            use_custom_allreduce=False,
            group_name="moe_tp",
            recovered_rank=recovered_rank,
        )

    num_pipeline_model_parallel_groups = (
        distributed_world_size // pipeline_model_parallel_size
    )
    assert (
        parallel_state._PP is None
    ), "pipeline model parallel group is already initialized"
    group_ranks = []
    for pp_group_idx in range(num_pipeline_model_parallel_groups):
        ranks = list(
            range(
                pp_group_idx, distributed_world_size, num_pipeline_model_parallel_groups
            )
        )
        group_ranks.append(ranks)
    parallel_state._PP = init_model_parallel_group(
        group_ranks,
        parallel_state._WORLD.local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="pp",
        recovered_rank=recovered_rank,
    )


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    """Initialize data parallel attention."""
    import sglang.srt.layers.dp_attention as dp_attention

    enable_dp_attention = server_args.enable_dp_attention
    tp_size = server_args.tp_size
    dp_size = server_args.dp_size
    moe_dense_tp_size = server_args.moe_dense_tp_size
    attn_cp_size = getattr(server_args, "attn_cp_size", 1)

    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    dp_attention._ENABLE_DP_ATTENTION_FLAG = enable_dp_attention

    (
        dp_attention._ATTN_TP_RANK,
        dp_attention._ATTN_TP_SIZE,
        dp_attention._ATTN_DP_RANK,
    ) = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, attn_cp_size
    )
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

    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )
