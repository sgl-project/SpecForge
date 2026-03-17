import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.core.dflash import OnlineDFlashModel, create_dflash_block_mask
from specforge.distributed import destroy_distributed, get_draft_sp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from tests.utils import get_available_port


class ManualAnchorOnlineDFlashModel(OnlineDFlashModel):
    def __init__(
        self,
        *,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id,
        block_size,
        attention_backend,
        anchor_positions,
        block_keep_mask,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=anchor_positions.shape[1],
        )
        self.manual_anchor_positions = anchor_positions
        self.manual_block_keep_mask = block_keep_mask

    def _sample_anchor_positions(self, seq_len, loss_mask, device):
        return (
            self.manual_anchor_positions.to(device),
            self.manual_block_keep_mask.to(device),
        )

    def _sample_anchor_positions_usp(self, position_ids, loss_mask, device):
        return (
            self.manual_anchor_positions.to(device),
            self.manual_block_keep_mask.to(device),
        )


def build_model_components(
    *,
    attention_backend: str,
    block_size: int,
    mask_token_id: int,
    hidden_size: int = 32,
    vocab_size: int = 256,
):
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        block_size=block_size,
        num_target_layers=1,
        rope_theta=10000.0,
        layer_types=["full_attention"],
    )
    config._attn_implementation = "flex_attention"
    config.dflash_config = {
        "target_layer_ids": [0],
        "mask_token_id": mask_token_id,
        "attention_backend": attention_backend,
    }
    draft_model = DFlashDraftModel(config).cuda().to(torch.float32).eval()
    embed_tokens = torch.nn.Embedding(vocab_size, hidden_size).cuda().to(torch.float32)
    lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False).cuda().to(torch.float32)
    return draft_model, embed_tokens, lm_head


def manual_forward_debug(
    model: OnlineDFlashModel,
    *,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    debug_capture: dict[str, torch.Tensor] | None = None,
):
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    use_usp_sampling = (
        position_ids is not None
        and dist.is_initialized()
        and get_draft_sp_group() is not None
        and dist.get_world_size(get_draft_sp_group()) > 1
    )
    if use_usp_sampling:
        global_anchor_positions, global_block_keep_mask = model._sample_anchor_positions_usp(
            position_ids, loss_mask, device
        )
        anchor_positions, block_keep_mask = model._slice_local_blocks(
            global_anchor_positions, global_block_keep_mask
        )
        real_context_len, padded_context_len = model._get_global_context_lengths(
            position_ids, attention_mask
        )
        anchor_tokens, _, anchor_presence = model._collect_global_values(
            anchor_positions, input_ids, loss_mask, position_ids, attention_mask
        )
        if not bool((anchor_presence | ~block_keep_mask).all().item()):
            raise ValueError("Failed to gather anchor tokens for USP DFlash")
        noise_embedding = model._create_noise_embed_from_anchor_tokens(
            anchor_tokens, block_keep_mask
        )
    else:
        anchor_positions, block_keep_mask = model._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        global_anchor_positions = anchor_positions
        global_block_keep_mask = block_keep_mask
        real_context_len = seq_len
        padded_context_len = seq_len
        noise_embedding = model._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

    if position_ids is not None:
        context_position_ids = position_ids[:, :seq_len]
    else:
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    draft_position_ids = model._create_position_ids(anchor_positions)
    full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

    dflash_attn_mask = create_dflash_block_mask(
        anchor_positions=global_anchor_positions,
        block_keep_mask=global_block_keep_mask,
        S=padded_context_len,
        block_size=model.block_size,
        device=device,
        real_context_len=real_context_len,
    )

    output_hidden = model.draft_model(
        position_ids=full_position_ids,
        noise_embedding=noise_embedding,
        target_hidden=hidden_states,
        attention_mask=dflash_attn_mask,
        debug_capture=debug_capture,
    )
    logits = model.lm_head(output_hidden)

    label_offsets = torch.arange(0, model.block_size, device=device).view(1, 1, -1)
    if use_usp_sampling:
        global_label_positions = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = global_label_positions < real_context_len
        target_ids, original_loss_mask_gathered, _ = model._collect_global_values(
            global_label_positions,
            input_ids,
            loss_mask,
            position_ids,
            attention_mask,
        )
    else:
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

    weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, model.block_size).float()
    weight_mask = weight_mask * valid_label_mask.float()
    pos_in_block = torch.arange(model.block_size, device=device).view(1, 1, -1)
    weight_mask = weight_mask * (pos_in_block > 0).float()
    weight_mask = weight_mask * original_loss_mask_gathered

    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = target_ids.view(-1)
    flat_weights = weight_mask.view(-1)
    loss_per_token = torch.nn.functional.cross_entropy(
        flat_logits, flat_targets, reduction="none"
    )
    loss_num = (loss_per_token * flat_weights).sum()
    denom = flat_weights.sum()
    if use_usp_sampling:
        sp_group = get_draft_sp_group()
        dist.all_reduce(loss_num, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(denom, op=dist.ReduceOp.SUM, group=sp_group)
    loss = loss_num / (denom + 1e-6)

    pred_ids = torch.argmax(flat_logits, dim=-1)
    correct = (pred_ids == flat_targets) & (flat_weights > 0.5)
    correct_num = correct.sum().float()
    correct_denom = (flat_weights > 0.5).sum().float()
    if use_usp_sampling:
        sp_group = get_draft_sp_group()
        dist.all_reduce(correct_num, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(correct_denom, op=dist.ReduceOp.SUM, group=sp_group)
    accuracy = correct_num / (correct_denom + 1e-6)

    return output_hidden, logits, loss, accuracy


def gather_seq_tensor(tensor: torch.Tensor, group) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=1)


def gather_tensor_dim(tensor: torch.Tensor, group, dim: int) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=dim)


def assert_close_with_name(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
):
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f"{name} mismatch\n{exc}") from exc


def setup_env(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)


def run_dflash_usp_parity(rank, world_size, port):
    setup_env(rank, world_size, port)
    set_seed(1234)
    device = torch.device(f"cuda:{rank}")

    block_size = 4
    mask_token_id = 99
    anchor_positions = torch.tensor([[2, 6]], dtype=torch.long, device=device)
    block_keep_mask = torch.tensor([[True, True]], dtype=torch.bool, device=device)
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]], device=device)
    loss_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32, device=device
    )
    hidden_size = 32
    hidden_states = torch.randn(1, input_ids.shape[1], hidden_size, device=device)

    baseline_draft_model, baseline_embed_tokens, baseline_lm_head = build_model_components(
        attention_backend="flex_attention",
        block_size=block_size,
        mask_token_id=mask_token_id,
        hidden_size=hidden_size,
    )
    baseline_wrapper = ManualAnchorOnlineDFlashModel(
        draft_model=baseline_draft_model,
        target_lm_head=baseline_lm_head,
        target_embed_tokens=baseline_embed_tokens,
        mask_token_id=mask_token_id,
        block_size=block_size,
        attention_backend="flex_attention",
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
    ).cuda().to(torch.float32)
    baseline_wrapper.eval()

    baseline_debug = {}
    with torch.no_grad():
        baseline_output, baseline_logits, baseline_loss, baseline_acc = manual_forward_debug(
            baseline_wrapper,
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            debug_capture=baseline_debug,
        )

    draft_state = baseline_draft_model.state_dict()
    embed_state = baseline_embed_tokens.state_dict()
    lm_head_state = baseline_lm_head.state_dict()

    init_distributed(tp_size=1, sp_ulysses_size=2, sp_ring_size=1)
    try:
        usp_draft_model, usp_embed_tokens, usp_lm_head = build_model_components(
            attention_backend="usp",
            block_size=block_size,
            mask_token_id=mask_token_id,
            hidden_size=hidden_size,
        )
        usp_draft_model.load_state_dict(draft_state)
        usp_embed_tokens.load_state_dict(embed_state)
        usp_lm_head.load_state_dict(lm_head_state)

        usp_wrapper = ManualAnchorOnlineDFlashModel(
            draft_model=usp_draft_model,
            target_lm_head=usp_lm_head,
            target_embed_tokens=usp_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend="usp",
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        ).cuda().to(torch.float32)
        usp_wrapper.eval()

        chunk_size = input_ids.shape[1] // world_size
        start = rank * chunk_size
        end = start + chunk_size
        local_input_ids = input_ids[:, start:end].contiguous()
        local_loss_mask = loss_mask[:, start:end].contiguous()
        local_hidden_states = hidden_states[:, start:end, :].contiguous()
        local_position_ids = torch.arange(start, end, device=device).unsqueeze(0)
        local_attention_mask = torch.ones_like(local_input_ids, dtype=torch.long)

        usp_debug = {}
        with torch.no_grad():
            usp_output, usp_logits, usp_loss, usp_acc = manual_forward_debug(
                usp_wrapper,
                input_ids=local_input_ids,
                hidden_states=local_hidden_states,
                loss_mask=local_loss_mask,
                position_ids=local_position_ids,
                attention_mask=local_attention_mask,
                debug_capture=usp_debug,
            )

        sp_group = get_draft_sp_group()
        usp_output_full = gather_seq_tensor(usp_output, sp_group)
        usp_logits_full = gather_seq_tensor(usp_logits, sp_group)

        gathered_debug = {
            "q_proj": gather_seq_tensor(usp_debug["q_proj"], sp_group),
            "k_ctx_proj": gather_seq_tensor(usp_debug["k_ctx_proj"], sp_group),
            "k_noise_proj": gather_seq_tensor(usp_debug["k_noise_proj"], sp_group),
            "v_ctx_proj": gather_seq_tensor(usp_debug["v_ctx_proj"], sp_group),
            "v_noise_proj": gather_seq_tensor(usp_debug["v_noise_proj"], sp_group),
            "q_after_rope": gather_tensor_dim(usp_debug["q_after_rope"], sp_group, 2),
            "k_after_rope": gather_tensor_dim(usp_debug["k_after_rope"], sp_group, 2),
            "q_scattered": gather_tensor_dim(usp_debug["q_scattered"], sp_group, 2),
            "k_ctx_scattered": gather_tensor_dim(
                usp_debug["k_ctx_scattered"], sp_group, 2
            ),
            "k_noise_scattered": gather_tensor_dim(
                usp_debug["k_noise_scattered"], sp_group, 2
            ),
            "v_ctx_scattered": gather_tensor_dim(
                usp_debug["v_ctx_scattered"], sp_group, 2
            ),
            "v_noise_scattered": gather_tensor_dim(
                usp_debug["v_noise_scattered"], sp_group, 2
            ),
            "attn_output_pre_gather": gather_tensor_dim(
                usp_debug["attn_output_pre_gather"], sp_group, 2
            ),
            "attn_output_final": gather_seq_tensor(usp_debug["attn_output_final"], sp_group),
            "attn_output_projected": gather_seq_tensor(
                usp_debug["attn_output_projected"], sp_group
            ),
        }

        if rank == 0:
            assert_close_with_name("q_proj", gathered_debug["q_proj"], baseline_debug["q_proj"])
            assert_close_with_name(
                "k_ctx_proj", gathered_debug["k_ctx_proj"], baseline_debug["k_ctx_proj"]
            )
            assert_close_with_name(
                "k_noise_proj",
                gathered_debug["k_noise_proj"],
                baseline_debug["k_noise_proj"],
            )
            assert_close_with_name(
                "v_ctx_proj", gathered_debug["v_ctx_proj"], baseline_debug["v_ctx_proj"]
            )
            assert_close_with_name(
                "v_noise_proj",
                gathered_debug["v_noise_proj"],
                baseline_debug["v_noise_proj"],
            )
            assert_close_with_name(
                "q_after_rope", gathered_debug["q_after_rope"], baseline_debug["q_after_rope"]
            )
            assert_close_with_name(
                "k_after_rope", gathered_debug["k_after_rope"], baseline_debug["k_after_rope"]
            )
            assert_close_with_name(
                "k_ctx_scattered",
                gathered_debug["k_ctx_scattered"],
                baseline_debug["k_after_rope"][:, :, : input_ids.shape[1], :].transpose(1, 2),
            )
            assert_close_with_name(
                "k_noise_scattered",
                gathered_debug["k_noise_scattered"],
                baseline_debug["k_after_rope"][:, :, input_ids.shape[1] :, :].transpose(1, 2),
            )
            assert_close_with_name(
                "v_ctx_scattered",
                gathered_debug["v_ctx_scattered"],
                baseline_debug["v_ctx_proj"],
            )
            assert_close_with_name(
                "v_noise_scattered",
                gathered_debug["v_noise_scattered"],
                baseline_debug["v_noise_proj"],
            )
            assert_close_with_name(
                "q_scattered",
                gathered_debug["q_scattered"],
                baseline_debug["q_after_rope"].transpose(1, 2),
            )
            assert_close_with_name(
                "attn_output_pre_gather",
                gathered_debug["attn_output_pre_gather"],
                baseline_debug["attn_output_pre_gather"],
            )
            assert_close_with_name(
                "attn_output_final",
                gathered_debug["attn_output_final"],
                baseline_debug["attn_output_final"],
            )
            assert_close_with_name(
                "attn_output_projected",
                gathered_debug["attn_output_projected"],
                baseline_debug["attn_output_projected"],
            )
            assert_close_with_name("output_hidden", usp_output_full, baseline_output)
            assert_close_with_name("logits", usp_logits_full, baseline_logits)
            assert_close_with_name("loss", usp_loss, baseline_loss, atol=1e-5, rtol=1e-5)
            assert_close_with_name("accuracy", usp_acc, baseline_acc, atol=1e-5, rtol=1e-5)
    finally:
        destroy_distributed()


class TestDFlashUSPParity(unittest.TestCase):
    def test_dflash_usp_matches_baseline(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(run_dflash_usp_parity, nprocs=world_size, args=(world_size, port))


if __name__ == "__main__":
    unittest.main()
