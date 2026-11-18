# coding=utf-8
"""Shared tiny synthetic fixtures for canonical runtime GPU tests.

Not a test module (no ``test_`` prefix). Builds a tiny EAGLE3 draft model, a
``TargetHead``, a vocab mapping, and offline ``.ckpt`` feature files — all from
random tensors, with NO model download. Offline batches are materialized through
the same manifest-reader and feature-loader path used by training.
"""

import json
import os

import torch

H = 64
V = 256
D = 64


def tiny_config(
    kind="eagle3",
    *,
    hidden=H,
    vocab=V,
    draft_layers=1,
    block_size=4,
    mask_token_id=0,
    markov_rank=8,
):
    from dataclasses import replace

    from tests.test_config.draft_fixtures import draft_config
    from tests.test_config.test_draft_eagle3_spec import eagle_config
    from tests.test_modeling.test_canonical_domino import model_config as domino_config
    from tests.test_modeling.test_canonical_dspark import model_config as dspark_config

    config = {
        "eagle3": eagle_config,
        "dflash": draft_config,
        "domino": domino_config,
        "dspark": dspark_config,
    }[kind]()
    spec = config.spec_config
    features = replace(
        spec.target_features,
        target_layer_ids=(0, 1, 2) if kind == "eagle3" else (1,),
        hidden_size=hidden,
    )
    if kind != "eagle3":
        features = replace(features, mask_token_id=mask_token_id)
        spec = replace(spec, block=replace(spec.block, block_size=block_size))
    if kind == "domino":
        spec = replace(
            spec, gru=replace(spec.gru, hidden_size=hidden // 2, projection_size=hidden)
        )
    if kind == "dspark":
        spec = replace(
            spec, markov=replace(spec.markov, type="vanilla", rank=markov_rank)
        )
    return replace(
        config,
        hidden_size=hidden,
        vocab_size=vocab,
        draft_vocab_size=D if kind == "eagle3" else vocab,
        max_position_embeddings=512,
        target_bindings=(),
        target_logit_transform=None,
        layer_types=(config.layer_types[0],) * draft_layers,
        attention_configs={
            name: replace(node, head_dim=hidden // 4)
            for name, node in config.attention_configs.items()
        },
        mlp_configs={
            name: replace(node, intermediate_size=hidden * 2)
            for name, node in config.mlp_configs.items()
        },
        spec_config=replace(spec, target_features=features),
    )


def build_strategy(model, target_head=None, **kwargs):
    from torch.nn.parallel import DistributedDataParallel

    from specforge.training.strategies.canonical import CanonicalTrainStrategy

    runtime = model.module if isinstance(model, DistributedDataParallel) else model
    return CanonicalTrainStrategy(
        model,
        name=runtime.objective_name,
        objective=runtime.objective_name,
        required_features=runtime.registration.inputs,
    )


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

    init_distributed(timeout=10, tp_size=1)


def init_rank_distributed(
    rank,
    world_size,
    *,
    tp_size=1,
    sp_ulysses_size=1,
    sp_ring_size=1,
    port="29571",
):
    """Initialize one rank with the canonical target/draft topology.

    Defaults preserve the world-sized FSDP/DP layout used by the two-rank
    runtime tests. The explicit TP/SP arguments let the numerical parity gate
    exercise production trainer-TP1 x SP2 groups without a second initializer.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = port
    os.environ["SPECFORGE_DEVICE"] = "cuda"
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
        from specforge.config.draft.codec import encode_config

        json.dump(encode_config(tiny_config()), f)
    return path


def write_target_head_dir(d, hidden=H, vocab=V):
    from safetensors.torch import save_file

    os.makedirs(d, exist_ok=True)
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 128,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    save_file(
        {"lm_head.weight": torch.randn(vocab, hidden, dtype=torch.float32)},
        os.path.join(d, "model.safetensors"),
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


def build_offline_eagle3_loader(
    hidden_states_path: str,
    *,
    batch_size: int,
    run_id: str = "data",
    ttt_length: int = 3,
    max_len: int = 512,
    ref_slice: slice | None = None,
):
    """Build the canonical fixed-ref EAGLE3 loader used by runtime tests."""
    from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore
    from tests.test_algorithms.fixtures import builtin_algorithm_registry

    algorithm = builtin_algorithm_registry().resolve("eagle3")
    provider = algorithm.providers.offline_for("text")
    reader = provider.build_reader(
        hidden_states_path,
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
    )
    refs = reader.read()
    if ref_slice is not None:
        refs = refs[ref_slice]
    return FeatureDataLoader(
        LocalFeatureStore(f"{run_id}-features"),
        refs=refs,
        batch_size=batch_size,
        collate_fn=provider.build_collator(),
        per_sample_transform=provider.build_normalizer(max_len),
        strategy=algorithm.name,
    )


def load_eagle3(
    config_path, target_dir, vocab_path, *, ttt=3, backend="sdpa", **options
):
    from specforge.algorithms.objectives import build_objective_model
    from specforge.config.draft.codec import decode_config
    from specforge.modeling import build_draft_model
    from specforge.modeling.target.target_head import TargetHead

    with open(config_path) as stream:
        config = decode_config(json.load(stream))
    draft = build_draft_model(
        config, backend="sdpa" if backend in ("fa", "usp") else backend
    ).to(device="cuda", dtype=torch.bfloat16)
    draft.vocabulary.load_mapping(**torch.load(vocab_path, weights_only=True))
    draft.embed_tokens.requires_grad_(False)
    head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")
    model = build_objective_model(
        draft, target_head=head.fc, options=dict(ttt_length=ttt, **options)
    )
    model.attention_backend = backend
    if backend == "usp":
        from specforge.training.sequence_parallel import configure_sequence_parallel

        configure_sequence_parallel(model)
    return model.cuda()


def build_eagle3(workdir, ttt=3):
    cfg = write_draft_config(os.path.join(workdir, "draft.json"))
    target = write_target_head_dir(os.path.join(workdir, "target"))
    vocab = write_vocab_mapping(os.path.join(workdir, "vocab_mapping.pt"))
    model = load_eagle3(cfg, target, vocab, ttt=ttt)
    return model, model.target_head


# --- DFlash fixtures ---------------------------------------------------------
# DFlash has its own feature schema: ``hidden_states`` is the concatenated
# target-layer capture (no EAGLE3 aux/target swap or target distribution).


def write_offline_files_dflash(d, n=4, seq=32, hidden=H, vocab=V, seed=0):
    """Write synthetic DFlash/Domino offline feature files."""
    os.makedirs(d, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    for index in range(n):
        torch.save(
            {
                "input_ids": torch.randint(0, vocab, (seq,), generator=generator),
                "loss_mask": torch.ones(seq, dtype=torch.long),
                "hidden_states": torch.randn(1, seq, hidden, generator=generator).to(
                    torch.bfloat16
                ),
            },
            os.path.join(d, f"{index:04d}.ckpt"),
        )
    return d


def write_offline_files_dspark(
    d,
    n=4,
    seq=32,
    captured_width=H,
    target_hidden=H,
    vocab=V,
    seed=0,
):
    """Write synthetic DSpark offline target-layer and final-state features."""

    os.makedirs(d, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    for index in range(n):
        torch.save(
            {
                "input_ids": torch.randint(0, vocab, (seq,), generator=generator),
                "loss_mask": torch.ones(seq, dtype=torch.long),
                "hidden_states": torch.randn(
                    1,
                    seq,
                    captured_width,
                    generator=generator,
                ).to(torch.bfloat16),
                "target_last_hidden_states": torch.randn(
                    1,
                    seq,
                    target_hidden,
                    generator=generator,
                ).to(torch.bfloat16),
            },
            os.path.join(d, f"{index:04d}.ckpt"),
        )
    return d


def build_block_model(
    workdir,
    kind,
    *,
    hidden=H,
    vocab=V,
    target_layers=4,
    draft_layers=1,
    block_size=4,
    num_anchors=8,
    mask_token_id=0,
    attention_backend="sdpa",
    markov_rank=8,
):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from specforge.algorithms.objectives import build_objective_model
    from specforge.modeling import build_draft_model
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    target_config = Qwen3Config(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=target_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        vocab_size=vocab,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    torch.manual_seed(1234)
    target_dir = os.path.join(workdir, kind + "_target")
    Qwen3ForCausalLM(target_config).save_pretrained(target_dir)
    config = tiny_config(
        kind,
        hidden=hidden,
        vocab=vocab,
        draft_layers=draft_layers,
        block_size=block_size,
        mask_token_id=mask_token_id,
        markov_rank=markov_rank,
    )
    draft = build_draft_model(config, backend=attention_backend).to(
        device="cuda", dtype=torch.bfloat16
    )
    target = TargetEmbeddingsAndHead.from_pretrained(
        target_dir, lm_head_key="lm_head.weight", device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(target.embed_tokens.weight)
        draft.lm_head.weight.copy_(target.lm_head.weight)
    draft.embed_tokens.requires_grad_(False)
    draft.lm_head.requires_grad_(False)
    if hasattr(draft, "initialize_transfer_space"):
        draft.initialize_transfer_space()
    options = dict(num_anchors=num_anchors)
    model = build_objective_model(
        draft, target_head=target.lm_head, options=options
    ).cuda()
    layers = list(config.spec_config.target_features.target_layer_ids)
    return model, len(layers) * hidden, target_dir, layers


def build_dflash(workdir, **options):
    return build_block_model(workdir, "dflash", **options)


def build_domino(workdir, **options):
    return build_block_model(workdir, "domino", **options)


def build_dspark(workdir, **options):
    return build_block_model(workdir, "dspark", **options)[:2]
