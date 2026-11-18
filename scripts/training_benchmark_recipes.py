"""Deterministic inputs for the native TorchTitan/FSDP training benchmark.

This module contains recipe and feature construction only, not a backend or
training loop. Keep heavy framework imports inside functions so the recipe
contract tests can run on CPU-only development hosts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


STOCK_CONFIGS = {
    "dflash": "qwen3-4b-dflash.json",
    "dflash2": "qwen3.5-4b-dflash2.json",
    "dspark": "qwen3-4b-dspark.json",
}


ARCHITECTURES = {
    "dflash": "DFlashDraftModel",
    "dflash2": "DFlash2DraftModel",
    "dspark": "DSparkDraftModel",
}


def resolve_config(algorithm, recipe="controlled", *, tiny=False, config_path=None):
    """Resolve a fully recorded recipe without importing GPU dependencies."""
    source = (
        Path(config_path)
        if config_path
        else REPO_ROOT
        / "configs"
        / (STOCK_CONFIGS[algorithm] if recipe == "stock" else STOCK_CONFIGS["dflash"])
    )
    config = json.loads(source.read_text())
    if config_path and config.get("architectures") != [ARCHITECTURES[algorithm]]:
        raise ValueError("--config architecture does not match --algorithm")
    if not config_path and recipe == "controlled":
        config["architectures"] = [ARCHITECTURES[algorithm]]
        config.pop("auto_map", None)
        method = config["dflash_config"]
        if algorithm == "dflash2":
            method.update(
                conv_group_size=16,
                conv_kernel_size=2,
                selector_rank=256,
                selector_top_k=16,
            )
        elif algorithm == "dspark":
            method.update(
                attention_mode="gqa",
                projector_type="dspark",
                markov_head_type="vanilla",
                markov_rank=256,
                confidence_head_alpha=1.0,
                enable_confidence_head=True,
                confidence_head_with_markov=True,
            )
    if tiny:
        config.update(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            head_dim=8,
            num_target_layers=4,
            vocab_size=128,
            max_position_embeddings=4096,
            layer_types=["full_attention"] * 2,
            block_size=4,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            rope_scaling=None,
            rope_theta=10000.0,
        )
        config.pop("rope_parameters", None)
        method = config["dflash_config"]
        method.update(target_layer_ids=[1, 2], mask_token_id=127, block_size=4)
        if algorithm == "dflash2":
            method.update(conv_group_size=4, selector_rank=8, selector_top_k=4)
        if algorithm == "dspark":
            method["markov_rank"] = 8
    # Match AutoDraftModelConfig: a draft owns no target embedding/head to tie.
    config["tie_word_embeddings"] = False
    return config, str(source.resolve())


def summarize_times(seconds, input_tokens_per_window):
    if not seconds or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError("step durations must be non-empty, finite and positive")
    ordered = sorted(seconds)
    return {
        "optimizer_step_seconds_mean": statistics.mean(seconds),
        "optimizer_step_seconds_median": statistics.median(seconds),
        "optimizer_step_seconds_p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "input_context_tokens_per_second": input_tokens_per_window
        * len(seconds)
        / sum(seconds),
    }


def _tensor_fingerprint(values):
    """Exact content hash, evaluated outside timing before sharding."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


NONPERSISTENT_BUFFER_POLICY = "fresh-reference-fp32-v1"


def nonpersistent_buffers(model, *, clone=False):
    """Read buffers omitted by state_dict, retaining stable unwrapped names."""
    result = {}
    for module_name, module in model.named_modules():
        parts = [
            part
            for part in module_name.split(".")
            if part not in {"_fsdp_wrapped_module", "_orig_mod"}
        ]
        for name in sorted(module._non_persistent_buffers_set):
            value = module._buffers.get(name)
            if value is None:
                continue
            path = ".".join([part for part in parts if part] + [name])
            if path in result:
                raise ValueError(f"Duplicate normalized buffer name: {path}")
            result[path] = value.detach().cpu().clone() if clone else value
    return result


def restore_nonpersistent_buffers(model, reference):
    """Undo benchmark-only dtype conversions using fresh pre-cast values.

    Legacy production construction casts all buffers to BF16. For this
    controlled comparison only, preserve the fresh constructor's FP32 RoPE
    frequencies, as native Titan initialization does. Copying into an already
    BF16 tensor would round again, so restore the reference dtype as well.
    """
    live = nonpersistent_buffers(model)
    if set(live) != set(reference):
        raise ValueError("Nonpersistent buffer names differ from fresh reference")
    for name, value in reference.items():
        if live[name].shape != value.shape:
            raise ValueError(f"Nonpersistent buffer shape differs: {name}")
        module_name, _, leaf = name.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        module._buffers[leaf] = value.to(device=live[name].device).clone()


def nonpersistent_buffer_metadata(model):
    values = nonpersistent_buffers(model)
    return {
        "sha256": _tensor_fingerprint(values),
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_fingerprint({name: value}),
            }
            for name, value in sorted(values.items())
        },
    }


def _build_model(args, config, device, dtype):
    import torch
    from torch import nn
    from transformers import Qwen3Config

    from specforge.algorithms.common.dflash_family_model import (
        OnlineDFlashModel,
        OnlineDSparkModel,
    )
    from specforge.modeling.auto import AutoDraftModel

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        torch.manual_seed(args.seed)
        draft_config = Qwen3Config.from_dict(copy.deepcopy(config))
        draft_config._attn_implementation = args.attention
        # Keep a fresh reference before AutoDraftModel's uniform dtype cast
        # can round explicit FP32 nonpersistent buffers (notably HF RoPE).
        # Parameters still initialize in the requested default dtype, with
        # exactly the same seed and values as the original benchmark.
        draft = AutoDraftModel.from_config(draft_config)
        reference_buffers = nonpersistent_buffers(draft, clone=True)
        if args.activation_checkpointing:
            draft.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        # Independent seed makes the target identical across algorithms, too.
        torch.manual_seed(args.seed + 1)
        embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        nn.init.normal_(embedding.weight, std=0.02)
        nn.init.normal_(head.weight, std=0.02)
        embedding.requires_grad_(False)
        head.requires_grad_(False)
        teacher_hash = _tensor_fingerprint(
            {"embedding": embedding.weight, "head": head.weight}
        )
        kwargs = dict(
            draft_model=draft,
            target_lm_head=head,
            target_embed_tokens=embedding,
            mask_token_id=config["dflash_config"]["mask_token_id"],
            block_size=draft.block_size,
            attention_backend=args.attention,
            num_anchors=args.num_anchors,
            objective_chunk_blocks=args.objective_chunk_blocks,
        )
        if args.algorithm == "dspark":
            model = OnlineDSparkModel(**kwargs)
        else:
            model = OnlineDFlashModel(**kwargs, teacher_metrics=args.detailed_metrics)
        counts = {
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "frozen_parameters": sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            ),
        }
        model.to(device=device, dtype=dtype)
        restore_nonpersistent_buffers(draft, reference_buffers)
        draft_hash = _tensor_fingerprint(draft.state_dict())
        return (
            model,
            draft,
            counts,
            draft_hash,
            teacher_hash,
        )
    finally:
        torch.set_default_dtype(old_dtype)


def _make_batches(args, config, rank, device, dtype):
    import torch

    from specforge.runtime.contracts import TrainBatch

    capture_width = (
        len(config["dflash_config"]["target_layer_ids"]) * config["hidden_size"]
    )
    if args.feature_file:
        raw_batches = torch.load(
            args.feature_file.format(rank=rank), map_location="cpu", weights_only=True
        )
        if not isinstance(raw_batches, list) or not raw_batches:
            raise ValueError(
                "feature file must be a nonempty list of tensor dictionaries"
            )
    else:
        raw_batches = []
        for index in range(args.cache_batches):
            generator = torch.Generator().manual_seed(
                args.seed + 1000 + rank * 10000 + index
            )
            shape = (args.batch_size, args.seq_length)
            mask = torch.ones(shape, dtype=torch.float32)
            # Vary supervision by rank/cache entry, exercising token weighting.
            tail = (rank + index) % max(1, args.seq_length // 4)
            if tail:
                mask[:, -tail:] = 0
            raw_batches.append(
                {
                    "input_ids": torch.randint(
                        0, config["vocab_size"], shape, generator=generator
                    ),
                    "loss_mask": mask,
                    "hidden_states": torch.randn(
                        *shape, capture_width, generator=generator, dtype=dtype
                    ),
                    "target_last_hidden_states": torch.randn(
                        *shape, config["hidden_size"], generator=generator, dtype=dtype
                    ),
                }
            )
    expected = {
        "input_ids": (args.batch_size, args.seq_length),
        "loss_mask": (args.batch_size, args.seq_length),
        "hidden_states": (args.batch_size, args.seq_length, capture_width),
        "target_last_hidden_states": (
            args.batch_size,
            args.seq_length,
            config["hidden_size"],
        ),
    }
    batches, fingerprints = [], []
    cache_bytes = 0
    for index, tensors in enumerate(raw_batches):
        if not isinstance(tensors, dict) or any(
            name not in tensors for name in expected
        ):
            raise ValueError(f"batch {index} must contain {sorted(expected)}")
        for name, shape in expected.items():
            if (
                not isinstance(tensors[name], torch.Tensor)
                or tuple(tensors[name].shape) != shape
            ):
                raise ValueError(f"batch {index} {name} must have shape {shape}")
        if (
            tensors["input_ids"].min() < 0
            or tensors["input_ids"].max() >= config["vocab_size"]
        ):
            raise ValueError("feature token id outside target vocabulary")
        normalized = {
            name: tensor.to(
                dtype=(
                    torch.long
                    if name == "input_ids"
                    else torch.float32 if name == "loss_mask" else dtype
                )
            )
            for name, tensor in tensors.items()
            if name in expected
        }
        fingerprints.append(_tensor_fingerprint(normalized))
        # Match prepositioned feature consumers: hidden states on GPU; small
        # integer/mask inputs on pinned host memory for CPU anchor counting.
        for name in normalized:
            if name in ("input_ids", "loss_mask"):
                normalized[name] = normalized[name].pin_memory()
            else:
                normalized[name] = normalized[name].to(device)
                cache_bytes += (
                    normalized[name].numel() * normalized[name].element_size()
                )
        batches.append(
            TrainBatch(
                sample_ids=[
                    f"rank{rank}-cache{index}-sample{n}" for n in range(args.batch_size)
                ],
                strategy="dspark" if args.algorithm == "dspark" else "dflash",
                tensors=normalized,
            )
        )
    return batches, fingerprints, cache_bytes
