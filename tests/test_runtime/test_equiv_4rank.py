# coding=utf-8
"""Four-rank DP2 x SP2/USP parity for the unified Trainer lifecycle.

The distributed leg uses the current offline builder and no-argument
``Trainer.fit()`` with trainer TP1, Ulysses-SP2 and world-sized FSDP groups. Its
logged boundary loss is compared with the same initial model and samples on the
canonical full-sequence flex-attention reference.  This replaces the deleted
legacy-script differential while retaining the scale-out numerical gate.
"""

import json
import math
import os
import tempfile
import unittest

import torch

from tests.test_runtime import _fixtures as fx

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD_SIZE = 4


def _has_standard_flash_attention() -> bool:
    """Probe the exact standard interfaces used by the USP implementation."""
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn.bert_padding import pad_input, unpad_input  # noqa: F401
        from flash_attn.flash_attn_interface import (  # noqa: F401
            _flash_attn_varlen_backward,
        )
    except Exception:
        return False
    return True


def _build_model(workdir: str, attention_backend: str):
    from specforge.algorithms.eagle3.model import OnlineEagle3Model
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

    draft_config = AutoDraftModelConfig.from_file(os.path.join(workdir, "draft.json"))
    draft_model = AutoDraftModel.from_config(
        draft_config,
        attention_backend=attention_backend,
        torch_dtype=torch.bfloat16,
    ).cuda()
    draft_model.load_vocab_mapping(os.path.join(workdir, "vm.pt"))
    draft_model.freeze_embedding()
    return OnlineEagle3Model(
        draft_model=draft_model,
        length=3,
        attention_backend=attention_backend,
    ).cuda()


def _worker(rank: int, world_size: int, port: int, workdir: str) -> None:
    fx.init_rank_distributed(
        rank,
        world_size,
        tp_size=1,
        sp_ulysses_size=2,
        sp_ring_size=1,
        port=str(port),
    )
    try:
        import torch.distributed as dist

        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.distributed import get_draft_sp_group, get_tp_group
        from specforge.launch import _shard_offline_refs, build_offline_runtime
        from specforge.modeling.target.target_head import TargetHead
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True, warn_only=True)

        reference_model = _build_model(workdir, "flex_attention")
        usp_model = _build_model(workdir, "usp")
        usp_model.load_state_dict(reference_model.state_dict())
        target_head = TargetHead.from_pretrained(
            os.path.join(workdir, "target"), lm_head_key="lm_head.weight"
        )

        algorithm = builtin_algorithm_registry().resolve("eagle3")
        provider = algorithm.providers.offline_for("text")
        source_refs = provider.build_reader(
            os.path.join(workdir, "features"),
            run_id="usp-reference",
            ttt_length=3,
            max_len=512,
        ).read()
        rank_refs = _shard_offline_refs(
            source_refs,
            use_usp_preprocess=True,
            seed=0,
            epoch=0,
        )
        if len(rank_refs) != 2:
            raise AssertionError(
                f"rank {rank} expected two accumulation refs, got {len(rank_refs)}"
            )

        reference_loader = FeatureDataLoader(
            LocalFeatureStore(f"usp-reference-{rank}"),
            refs=[rank_refs[-1]],
            batch_size=1,
            collate_fn=provider.build_collator(),
            per_sample_transform=provider.build_normalizer(
                512,
                ttt_length=3,
                use_usp_preprocess=False,
            ),
            strategy=algorithm.name,
        )
        reference_batch = next(iter(reference_loader))
        reference_model.train()
        with torch.no_grad():
            expected_loss = (
                algorithm.providers.step.build(
                    reference_model,
                    target_head=target_head,
                )
                .forward_loss(reference_batch)
                .loss.detach()
            )
        dist.all_reduce(expected_loss, op=dist.ReduceOp.SUM)
        expected_loss /= world_size

        def optimizer_factory(module):
            return BF16Optimizer(
                module,
                lr=1e-4,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=1,
            )

        logged = []
        trainer = build_offline_runtime(
            algorithm=algorithm,
            hidden_states_path=os.path.join(workdir, "features"),
            draft_model=usp_model,
            target_head=target_head,
            optimizer_factory=optimizer_factory,
            run_id="usp-parity",
            output_dir=os.path.join(workdir, "output"),
            ttt_length=3,
            max_len=512,
            batch_size=1,
            # Production assembly multiplies one logical accumulation unit by
            # SP size; two refs per draft-DP shard form one optimizer boundary.
            accumulation_steps=2,
            num_epochs=1,
            max_steps=1,
            tp_size=1,
            sp_ulysses_size=2,
            sp_ring_size=1,
            use_usp_preprocess=True,
            seed=0,
            logger=lambda metrics, step: logged.append((step, dict(metrics))),
            log_interval=1,
        )
        step = trainer.fit()
        if len(logged) != 1:
            raise AssertionError(f"rank {rank} expected one logged step, got {logged}")

        topology = trainer.backend.parallel_config
        result = {
            "rank": rank,
            "step": step,
            "world_size": topology.world_size,
            "tp_size": topology.tp_size,
            "sp_size": topology.sp_size,
            "tp_group_size": dist.get_world_size(get_tp_group()),
            "draft_sp_group_size": dist.get_world_size(get_draft_sp_group()),
            "reference_loss": float(expected_loss.item()),
            "usp_loss": float(logged[0][1]["loss"]),
            "grad_norm": float(logged[0][1]["grad_norm"]),
        }
        with open(os.path.join(workdir, "results", f"rank{rank}.json"), "w") as out:
            json.dump(result, out)
    finally:
        from specforge.distributed import destroy_distributed

        destroy_distributed()


@unittest.skipUnless(
    CUDA and NGPU >= WORLD_SIZE and _has_standard_flash_attention(),
    "requires four CUDA devices and the standard flash-attn USP interfaces",
)
class TestEquiv4Rank(unittest.TestCase):
    def test_dp2_sp2_trainer_loss_matches_full_sequence_reference(self):
        import torch.multiprocessing as mp

        from tests.utils import get_available_port

        with tempfile.TemporaryDirectory(prefix="equiv_4rank_") as work:
            fx.write_draft_config(os.path.join(work, "draft.json"))
            fx.write_target_head_dir(os.path.join(work, "target"))
            fx.write_vocab_mapping(os.path.join(work, "vm.pt"))
            fx.write_offline_files(os.path.join(work, "features"), n=4, seq=64, seed=23)
            os.makedirs(os.path.join(work, "results"))

            mp.spawn(
                _worker,
                nprocs=WORLD_SIZE,
                args=(WORLD_SIZE, get_available_port(), work),
                join=True,
            )

            for rank in range(WORLD_SIZE):
                with open(os.path.join(work, "results", f"rank{rank}.json")) as src:
                    result = json.load(src)
                self.assertEqual(result["rank"], rank)
                self.assertEqual(result["step"], 1)
                self.assertEqual(result["world_size"], WORLD_SIZE)
                self.assertEqual(result["tp_size"], 1)
                self.assertEqual(result["sp_size"], 2)
                self.assertEqual(result["tp_group_size"], 1)
                self.assertEqual(result["draft_sp_group_size"], 2)
                self.assertTrue(
                    math.isclose(
                        result["reference_loss"],
                        result["usp_loss"],
                        rel_tol=2e-2,
                        abs_tol=2e-2,
                    ),
                    f"rank {rank} full/USP loss mismatch: {result}",
                )
                self.assertTrue(math.isfinite(result["grad_norm"]))
                self.assertGreater(result["grad_norm"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
