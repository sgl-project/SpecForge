"""Distributed numerical test for TorchTitan's DFlash tensor parallel model.

Run with ``torchrun --standalone --nproc-per-node=2 -m
tests.test_training.test_torchtitan_parallel --tp 2``.  Every rank compares the
distributed objective and reconstructed gradients with an unpartitioned model.
"""

from __future__ import annotations

import argparse
import json
import os


def main():
    # Compare the transparent objective/conv equations. Production-kernel
    # throughput is measured separately on representative vocabulary sizes.
    os.environ.setdefault("SPECFORGE_DFLASH_FUSED_HEAD", "0")
    os.environ.setdefault("SPECFORGE_DFLASH2_FUSED_CONV", "0")
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.distributed.utils import clip_grad_norm_, set_spmd_backend

    from specforge.training.torchtitan.model import SpecForgeTitanModel
    from specforge.training.torchtitan.parallelize import (
        HeterogeneousGradientNorms,
        group_optimizer_parameters_by_mesh,
        parallelize_dflash,
    )
    from specforge.training.torchtitan.runtime import partition_anchor_blocks

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--anchors", type=int, default=5)
    parser.add_argument("--algorithm", default="all")
    parser.add_argument("--loss-type", default="dflash")
    parser.add_argument("--lk-loss-type", default=None)
    parser.add_argument("--ac", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    set_spmd_backend("partial_dtensor")
    world = dist.get_world_size()
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=world // (args.tp * args.cp),
        tp=args.tp,
        cp=args.cp,
        pp=1,
        ep=1,
        world_size=world,
        spmd_backend="partial_dtensor",
    )
    dims.build_mesh()
    algorithms = (
        ("dflash", "dflash2", "dspark")
        if args.algorithm == "all"
        else (args.algorithm,)
    )
    results = []
    compute_dtype = torch.float32 if args.fp32 else torch.bfloat16
    for algorithm in algorithms:
        cfg = SpecForgeTitanModel.Config(
            algorithm=algorithm,
            teacher_dtype="float32" if args.fp32 else "bfloat16",
            draft_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "num_target_layers": 4,
                "head_dim": 8,
                "max_position_embeddings": 128,
                "vocab_size": 64,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "layer_types": ["full_attention", "full_attention"],
                "dflash_config": {
                    "block_size": 4,
                    "mask_token_id": 63,
                    "target_layer_ids": [1, 2],
                    "conv_group_size": 4,
                    "conv_kernel_size": 2,
                    "selector_rank": 4,
                    "selector_top_k": 4,
                    "markov_rank": 4,
                    "markov_head_type": "gated",
                    "enable_confidence_head": True,
                    "confidence_head_with_markov": True,
                },
            },
            objective={
                "attention_backend": "eager",
                "num_anchors": args.anchors,
                "objective_chunk_blocks": 0,
            },
        )
        if algorithm != "dspark":
            cfg.objective["loss_type"] = args.loss_type
            cfg.objective["lk_loss_type"] = args.lk_loss_type
        reference = cfg.build().cuda()
        reference.init_weights()
        with torch.device("meta"):
            distributed = cfg.build()
        parallelize_dflash(
            distributed,
            parallel_dims=dims,
            training=TrainingConfig(
                mixed_precision_param="float32" if args.fp32 else "bfloat16",
                mixed_precision_reduce="float32",
            ),
            parallelism=ParallelismConfig(
                spmd_backend="partial_dtensor",
                enable_sequence_parallel=False,
                tensor_parallel_degree=args.tp,
                context_parallel_degree=args.cp,
            ),
            compile_config=CompileConfig(enable=args.compile),
            ac_config=FullAC.Config() if args.ac else None,
            dump_folder="/tmp/specforge-titan-parity",
        )
        distributed.to_empty(device="cuda")
        distributed.init_weights()
        # SUM over DP is intentional in TorchTitan; compare to that objective.
        batch_group = dims.get_optional_mesh(
            "batch", include_singleton_axes=True
        ).get_group()
        reference.training_model.objective_process_group = batch_group
        reference.training_model.objective_context_group = batch_group
        generator = torch.Generator().manual_seed(143)
        batch = {
            "input_ids": torch.randint(0, 63, (1, 16), generator=generator).cuda(),
            "hidden_states": torch.randn(1, 16, 64, generator=generator)
            .cuda()
            .to(compute_dtype),
            "loss_mask": torch.ones(1, 16, dtype=torch.long).cuda(),
            "target_last_hidden_states": torch.randn(1, 16, 32, generator=generator)
            .cuda()
            .to(compute_dtype),
            "collect_detailed_metrics": False,
        }
        anchors = torch.arange(args.anchors, device="cuda").unsqueeze(0) * 3
        keep = torch.ones_like(anchors, dtype=torch.bool)
        batch["anchor_positions"] = anchors
        batch["block_keep_mask"] = keep
        context_mesh = dims.get_optional_mesh("cp", include_singleton_axes=True)
        local_anchors, local_keep = partition_anchor_blocks(
            anchors, keep, rank=context_mesh.get_local_rank(), degree=args.cp
        )
        local_batch = dict(
            batch, anchor_positions=local_anchors, block_keep_mask=local_keep
        )
        if algorithm != "dspark":
            normalizer = reference.training_model.prepared_objective_denominator(
                batch["loss_mask"], anchors, keep
            )
            if args.loss_type != "dflash":
                _, weights = reference.training_model._dflash_weight_mask(
                    batch["loss_mask"], anchors, keep
                )
                scale = reference.training_model._sequence_anchor_scale(
                    weights
                ).squeeze(-1)
                local_scale, _ = partition_anchor_blocks(
                    scale, keep, rank=context_mesh.get_local_rank(), degree=args.cp
                )
                local_batch["prepared_sequence_anchor_scale"] = local_scale.unsqueeze(
                    -1
                )
        losses = []
        for model, model_batch in ((reference, batch), (distributed, local_batch)):
            torch.cuda.manual_seed(735)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                loss, _, metrics = model(**model_batch)
                if algorithm != "dspark":
                    loss = metrics["loss_terms"][0] / normalizer
                elif model is distributed:
                    loss = loss / args.cp
            loss.backward()
            losses.append(loss.detach().float())
        dist.all_reduce(losses[1], group=context_mesh.get_group())
        torch.testing.assert_close(
            losses[0],
            losses[1],
            rtol=1e-5 if args.fp32 else 0.01,
            atol=1e-6 if args.fp32 else 0.005,
        )
        reference_parameters = dict(reference.named_parameters())
        errors = {}
        for name, parameter in distributed.named_parameters():
            name = name.replace("._checkpoint_wrapped_module", "")
            if not parameter.requires_grad:
                assert parameter.grad is None, name
                continue
            expected = reference_parameters[name].grad
            gradient = parameter.grad
            if expected is None or gradient is None:
                assert expected is None and gradient is None, name
                continue
            if isinstance(gradient, DTensor):
                gradient = gradient.full_tensor()
            expected = expected * (world // (args.tp * args.cp))
            torch.testing.assert_close(
                gradient.float(),
                expected.float(),
                rtol=5e-4 if args.fp32 else 0.05,
                atol=1e-6 if args.fp32 else 0.005,
                msg=name,
            )
            errors[name] = float((gradient.float() - expected.float()).abs().max())
        for parameter in reference.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(world // (args.tp * args.cp))
        reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25)
        with HeterogeneousGradientNorms():
            distributed_norm = clip_grad_norm_(
                distributed.parameters(), 0.25, foreach=True
            )
        torch.testing.assert_close(
            distributed_norm,
            reference_norm,
            rtol=1e-4 if args.fp32 else 0.02,
            atol=1e-6 if args.fp32 else 0.005,
        )
        optimizer = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.001, "weight_decay": 0.01},
                )
            ]
        ).build(model_parts=[distributed])
        group_optimizer_parameters_by_mesh(optimizer, [distributed], dims)
        reference_optimizer = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad],
            lr=0.001,
            weight_decay=0.01,
            fused=True,
        )
        before_step = {
            name: parameter.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        for name, parameter in distributed.named_parameters():
            name = name.replace("._checkpoint_wrapped_module", "")
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if isinstance(gradient, DTensor):
                gradient = gradient.full_tensor()
            torch.testing.assert_close(
                gradient,
                reference_parameters[name].grad,
                rtol=5e-4 if args.fp32 else 0.05,
                atol=1e-6 if args.fp32 else 0.001,
                msg=f"clipped gradient {name}",
            )
        reference_optimizer.step()
        optimizer.step()
        optimizer_state = optimizer.state_dict()
        saved_parameter_names = {
            key.removeprefix("state.").removesuffix(".exp_avg")
            for key in optimizer_state
            if key.startswith("state.") and key.endswith(".exp_avg")
        }
        assert saved_parameter_names == set(errors)
        actual_updates, reference_updates = [], []
        for name, parameter in distributed.named_parameters():
            name = name.replace("._checkpoint_wrapped_module", "")
            value = (
                parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            )
            if not parameter.requires_grad:
                torch.testing.assert_close(value, before_step[name], rtol=0, atol=0)
                continue
            actual_updates.append(
                (value - before_step[name]).detach().float().flatten()
            )
            reference_updates.append(
                (reference_parameters[name] - before_step[name])
                .detach()
                .float()
                .flatten()
            )
        actual_update = torch.cat(actual_updates)
        reference_update = torch.cat(reference_updates)
        update_error = (
            actual_update - reference_update
        ).norm() / reference_update.norm()
        update_cosine = torch.nn.functional.cosine_similarity(
            actual_update, reference_update, dim=0
        )
        # BF16 near-zero gradients can change Adam's first-step sign; require
        # agreement of the complete update, in addition to each gradient above.
        assert update_error < (0.02 if args.fp32 else 0.15), float(update_error)
        assert update_cosine > (0.999 if args.fp32 else 0.99), float(update_cosine)
        results.append(
            {
                "algorithm": algorithm,
                "tp": args.tp,
                "dp": world // (args.tp * args.cp),
                "cp": args.cp,
                "loss_type": args.loss_type if algorithm != "dspark" else "kl",
                "lk_loss_type": args.lk_loss_type if algorithm != "dspark" else None,
                "activation_checkpoint": args.ac,
                "compile": args.compile,
                "compute_dtype": str(compute_dtype),
                "fused_head": os.environ["SPECFORGE_DFLASH_FUSED_HEAD"] == "1",
                "fused_conv": os.environ["SPECFORGE_DFLASH2_FUSED_CONV"] == "1",
                "reference_loss": float(losses[0]),
                "distributed_loss": float(losses[1]),
                "max_gradient_abs_error": max(errors.values()),
                "checked_gradients": len(errors),
                "native_gradient_norm": float(distributed_norm),
                "native_optimizer_step": True,
                "optimizer_update_relative_error": float(update_error),
                "optimizer_update_cosine": float(update_cosine),
            }
        )
        del reference, distributed
        torch.cuda.empty_cache()
    if dist.get_rank() == 0:
        print(json.dumps(results, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
