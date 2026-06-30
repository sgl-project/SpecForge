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


def build_hf_target(workdir, hidden=H, layers=8, vocab=V, aux_layer_ids=(1, 3, 4)):
    """Build a tiny HF Llama target wrapped by the SpecForge HF eagle3 backend."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from specforge.modeling.target import get_eagle3_target_model

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
    target = get_eagle3_target_model(
        pretrained_model_name_or_path=target_dir,
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
