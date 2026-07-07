# coding=utf-8
"""Shared tiny synthetic fixtures for runtime GPU equivalence tests.

Not a test module (no ``test_`` prefix). Builds a tiny EAGLE3 draft model, a
``TargetHead``, a vocab mapping, and offline ``.ckpt`` feature files — all from
random tensors, with NO model download — so the differential-equivalence tests
compare the old vs new code path on identical inputs/weights.
"""

import json
import os

import torch

TINY_DRAFT_CONFIG = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128,
    "max_position_embeddings": 512,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 256,
    "draft_vocab_size": 64,
}

H = 64
V = 256
D = 64


def build_single_rank_distributed(port="29561"):
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", port)
    torch.cuda.set_device(0)
    from specforge.distributed import init_distributed

    init_distributed(timeout=10, tp_size=1, sp_ulysses_size=1, sp_ring_size=1)


def init_rank_distributed(
    rank,
    world_size,
    *,
    tp_size=1,
    sp_ulysses_size=1,
    sp_ring_size=1,
    port="29571",
):
    """Init one rank of a multi-process group (for the M6 >=4-rank tests).

    Each spawned process calls this with its own rank; together they form a
    world_size group with the requested TP x SP layout. Mirrors the env that
    torchrun sets, so SpecForge's ``init_distributed`` builds the real TP +
    Ulysses/Ring SP groups.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    from specforge.distributed import init_distributed

    init_distributed(
        timeout=60,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
    )


def write_draft_config(path):
    with open(path, "w") as f:
        json.dump(TINY_DRAFT_CONFIG, f)
    return path


def write_target_head_dir(d, hidden=H, vocab=V):
    from safetensors.torch import save_file

    os.makedirs(d, exist_ok=True)
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 128,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    save_file(
        {"lm_head.weight": torch.randn(vocab, hidden, dtype=torch.float32)},
        os.path.join(d, "model.safetensors"),
    )
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {}, "weight_map": {"lm_head.weight": "model.safetensors"}}, f
        )
    return d


def write_vocab_mapping(path, vocab=V, draft_vocab=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    draft_ids = torch.randperm(vocab, generator=g)[:draft_vocab].sort().values
    t2d = torch.zeros(vocab, dtype=torch.bool)
    t2d[draft_ids] = True
    d2t = (draft_ids - torch.arange(draft_vocab)).to(torch.int64)
    torch.save({"t2d": t2d, "d2t": d2t}, path)
    return path


def write_offline_files(d, n=4, seq=16, hidden=H, vocab=V, seed=0):
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        torch.save(
            {
                "input_ids": torch.randint(0, vocab, (seq,), generator=g),
                "loss_mask": torch.ones(seq, dtype=torch.long),
                # prepared offline features are bf16 (target ran in bf16)
                "hidden_state": torch.randn(1, seq, hidden, generator=g).to(
                    torch.bfloat16
                ),
                "aux_hidden_state": torch.randn(1, seq, 3 * hidden, generator=g).to(
                    torch.bfloat16
                ),
            },
            os.path.join(d, f"{i:04d}.ckpt"),
        )
    return d


TINY_MLA_DRAFT_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLMEagle3"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128,
    "max_position_embeddings": 512,
    "model_type": "deepseek_v3",
    "num_attention_heads": 4,
    "num_hidden_layers": 1,
    # MLA geometry (tiny): compressed KV + split nope/rope head dims.
    "q_lora_rank": 24,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "rope_scaling": None,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 256,
    "draft_vocab_size": 64,
}


def write_mla_draft_config(path):
    with open(path, "w") as f:
        json.dump(TINY_MLA_DRAFT_CONFIG, f)
    return path


def build_mla_eagle3(workdir, ttt=3):
    """(eagle3_model, target_head) with the MLA (DeepSeek) draft, on cuda.

    Mirrors :func:`build_eagle3`; the fixture dims (H/V/draft vocab) are shared,
    so the same offline feature files and vocab mapping drive both drafts —
    the algorithm surface is identical, only the draft ARCHITECTURE differs.
    MLA supports the sdpa backend (see deepseek_eagle3.py).
    """
    from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Model
    from specforge.modeling.target import TargetHead

    cfg = write_mla_draft_config(os.path.join(workdir, "mla_draft.json"))
    target_dir = write_target_head_dir(os.path.join(workdir, "target"))
    vocab_path = write_vocab_mapping(os.path.join(workdir, "vocab_mapping.pt"))

    draft_config = AutoDraftModelConfig.from_file(cfg)
    draft_model = AutoEagle3DraftModel.from_config(
        draft_config, attention_backend="sdpa", torch_dtype=torch.bfloat16
    ).cuda()
    draft_model.load_vocab_mapping(vocab_path)
    draft_model.freeze_embedding()
    target_head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")
    eagle3_model = OnlineEagle3Model(
        draft_model=draft_model, length=ttt, attention_backend="sdpa"
    ).cuda()
    return eagle3_model, target_head


def build_hf_target(workdir, hidden=H, layers=8, vocab=V, aux_layer_ids=(1, 3, 4)):
    """Build a tiny HF Llama target wrapped by the SpecForge HF eagle3 backend."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from specforge.modeling.target import get_target_engine

    cfg = LlamaConfig(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    torch.manual_seed(1234)
    model = LlamaForCausalLM(cfg)
    target_dir = os.path.join(workdir, "hf_target")
    model.save_pretrained(target_dir)
    target = get_target_engine(
        target_dir,
        strategy="eagle3",
        backend="hf",
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    target.set_aux_hidden_states_layers(list(aux_layer_ids))
    return target, target_dir, list(aux_layer_ids)


def build_eagle3(workdir, ttt=3):
    """Build (eagle3_model, target_head) sharing one set of weights, on cuda."""
    from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Model
    from specforge.modeling.target import TargetHead

    cfg = write_draft_config(os.path.join(workdir, "draft.json"))
    target_dir = write_target_head_dir(os.path.join(workdir, "target"))
    vocab_path = write_vocab_mapping(os.path.join(workdir, "vocab_mapping.pt"))

    draft_config = AutoDraftModelConfig.from_file(cfg)
    draft_model = AutoEagle3DraftModel.from_config(
        draft_config, attention_backend="flex_attention", torch_dtype=torch.bfloat16
    ).cuda()
    draft_model.load_vocab_mapping(vocab_path)
    draft_model.freeze_embedding()
    target_head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")
    eagle3_model = OnlineEagle3Model(
        draft_model=draft_model, length=ttt, attention_backend="flex_attention"
    ).cuda()
    return eagle3_model, target_head


# --- DFlash fixtures ---------------------------------------------------------
# DFlash has its OWN feature schema: 'hidden_states' = concat of the target
# capture layers (NO eagle3 aux/target swap, NO target distribution). For a
# single draft layer the capture set is one layer, so width == hidden_size.


def write_offline_files_dflash(d, n=4, seq=32, hidden=H, vocab=V, seed=0):
    """Write synthetic DFlash offline .ckpt files: {input_ids, loss_mask, hidden_states}.

    No production dumper for DFlash exists yet (prepare_hidden_states.py is the
    EAGLE3 dumper), so the offline DataFlow path is exercised with synthetic files.
    loss_mask is all-ones with seq >= 2*block_size so anchor sampling succeeds;
    hidden_states is bf16 (uniform dtype across files for the loader's spec check).
    """
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        torch.save(
            {
                "input_ids": torch.randint(0, vocab, (seq,), generator=g),
                "loss_mask": torch.ones(seq, dtype=torch.long),
                # width == len(target_layer_ids)*hidden; single draft layer -> hidden
                "hidden_states": torch.randn(1, seq, hidden, generator=g).to(
                    torch.bfloat16
                ),
            },
            os.path.join(d, f"{i:04d}.ckpt"),
        )
    return d


def build_dflash(
    workdir,
    *,
    hidden=H,
    vocab=V,
    target_layers=4,
    draft_layers=1,
    block_size=4,
    num_anchors=8,
    mask_token_id=0,
    attention_backend="sdpa",
):
    """Build a tiny OnlineDFlashModel on cuda, mirroring scripts/train_dflash.build_models.

    Returns (dflash_model, hidden_states_width, target_dir, target_layer_ids).
    target_dir holds the saved tiny Qwen3 target (load it as an HF DFlash target for
    the ONLINE path); target_layer_ids are the capture layers (== set_capture_layers).
    For draft_layers=1 the capture set is one target layer so width == hidden.
    """
    from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM

    from specforge.core.dflash import OnlineDFlashModel
    from specforge.modeling.draft.dflash import DFlashDraftModel
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    # tiny Qwen3 target saved to disk (draft config is derived from it, as in train_dflash)
    tcfg = Qwen3Config(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=target_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    torch.manual_seed(1234)
    target_dir = os.path.join(workdir, "dflash_target")
    Qwen3ForCausalLM(tcfg).save_pretrained(target_dir)

    draft_config = AutoConfig.from_pretrained(target_dir)
    draft_config.num_hidden_layers = draft_layers
    draft_config.block_size = block_size
    draft_config.num_target_layers = target_layers
    draft_config.dflash_config = {"mask_token_id": mask_token_id}
    draft_config._attn_implementation = attention_backend

    draft_model = DFlashDraftModel(draft_config).to(device="cuda", dtype=torch.bfloat16)
    draft_model.mask_token_id = mask_token_id

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        target_dir, lm_head_key="lm_head.weight", device="cuda", dtype=torch.bfloat16
    )

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=attention_backend,
        num_anchors=num_anchors,
        loss_type="dflash",
    ).cuda()
    width = len(draft_model.target_layer_ids) * hidden
    return dflash_model, width, target_dir, list(draft_model.target_layer_ids)


# --- DFlash MLA (DeepSeek) fixtures ------------------------------------------
# The draft-ARCHITECTURE axis for DFlash: the MLA draft (deepseek_dflash.py)
# trains through the SAME OnlineDFlashModel wrapper as the Qwen3-style draft.
# Building a real tiny DeepSeek target is heavy (MoE); the DFlash training
# forward only needs the target's embed/lm_head + captured features, so we use
# plain nn.Embedding / nn.Linear stand-ins and synthetic hidden states.

TINY_MLA_DFLASH_CONFIG = {
    "architectures": ["DeepseekDFlashDraftModel"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128,
    "max_position_embeddings": 512,
    "model_type": "deepseek_v3",
    "num_attention_heads": 4,
    # draft depth; target depth drives the auto capture-layer schedule.
    "num_hidden_layers": 2,
    "num_target_layers": 8,
    # MLA geometry (tiny): no q_lora (matches DeepSeek-V2-Lite), compressed KV,
    # split nope/rope head dims.
    "q_lora_rank": None,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "rope_scaling": None,
    "rope_theta": 10000,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 256,
    "draft_vocab_size": 256,
    "block_size": 4,
    "dflash_config": {"mask_token_id": 0},
}


def write_mla_dflash_config(path, **overrides):
    cfg = dict(TINY_MLA_DFLASH_CONFIG)
    cfg.update(overrides)
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def build_dflash_mla(
    workdir,
    *,
    vocab=256,
    num_anchors=8,
    mask_token_id=0,
    **config_overrides,
):
    """Build a tiny OnlineDFlashModel with the MLA (DeepSeek) draft, on cuda.

    Returns (dflash_model, hidden_states_width, target_layer_ids). Mirrors
    :func:`build_dflash` but swaps the draft architecture to the MLA draft and
    uses plain embed/lm_head stand-ins (no real DeepSeek target). MLA supports
    the sdpa training backend only.
    """
    from transformers import AutoConfig

    from specforge.core.dflash import OnlineDFlashModel
    from specforge.modeling.draft.deepseek_dflash import DeepseekDFlashDraftModel

    cfg_path = write_mla_dflash_config(
        os.path.join(workdir, "mla_dflash.json"), vocab_size=vocab, **config_overrides
    )
    draft_config = AutoConfig.from_pretrained(cfg_path)
    draft_config._attn_implementation = "sdpa"

    draft_model = DeepseekDFlashDraftModel(draft_config).to(
        device="cuda", dtype=torch.bfloat16
    )
    draft_model.mask_token_id = mask_token_id

    hidden = draft_config.hidden_size
    embed_tokens = torch.nn.Embedding(vocab, hidden).to(
        device="cuda", dtype=torch.bfloat16
    )
    lm_head = torch.nn.Linear(hidden, vocab, bias=False).to(
        device="cuda", dtype=torch.bfloat16
    )

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=lm_head,
        target_embed_tokens=embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_type="dflash",
    ).cuda()
    width = len(draft_model.target_layer_ids) * hidden
    return dflash_model, width, list(draft_model.target_layer_ids)


def build_domino(
    workdir,
    *,
    hidden=H,
    vocab=V,
    target_layers=4,
    draft_layers=1,
    block_size=4,
    num_anchors=8,
    mask_token_id=0,
    attention_backend="sdpa",
):
    """Build a tiny OnlineDominoModel on cuda, mirroring scripts/train_domino.

    Domino reuses the DFlash draft model with a projector_type="domino" head (GRU
    prefix + embed projection). Returns
    (domino_model, hidden_states_width, target_dir, target_layer_ids).
    """
    from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM

    from specforge.core.domino import OnlineDominoModel
    from specforge.modeling.draft.dflash import DFlashDraftModel
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    tcfg = Qwen3Config(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=target_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    torch.manual_seed(1234)
    target_dir = os.path.join(workdir, "domino_target")
    Qwen3ForCausalLM(tcfg).save_pretrained(target_dir)

    draft_config = AutoConfig.from_pretrained(target_dir)
    draft_config.num_hidden_layers = draft_layers
    draft_config.block_size = block_size
    draft_config.num_target_layers = target_layers
    draft_config.dflash_config = {
        "projector_type": "domino",  # builds the GRU prefix + embed_proj head
        "emb_dim": hidden,
        "gru_hidden_dim": hidden // 2,
        "pure_draft_prefix_len": 0,
        "shift_label": False,
        "mask_token_id": mask_token_id,
    }
    draft_config._attn_implementation = attention_backend

    draft_model = DFlashDraftModel(draft_config).to(device="cuda", dtype=torch.bfloat16)
    draft_model.mask_token_id = mask_token_id

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        target_dir, lm_head_key="lm_head.weight", device="cuda", dtype=torch.bfloat16
    )

    domino_model = OnlineDominoModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=attention_backend,
        num_anchors=num_anchors,
        shift_label=draft_model.shift_label,
    ).cuda()
    width = len(draft_model.target_layer_ids) * hidden
    return domino_model, width, target_dir, list(draft_model.target_layer_ids)
