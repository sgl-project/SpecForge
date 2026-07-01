# coding=utf-8
"""E1 gates: eval-cache identity, hit/produce-once semantics, eval-stream
equivalence (LocalFeatureStore vs SharedDirFeatureStore), and the loader-level
batch-size invariance gate.

The cache tests pin the §4.4 contract: every identity field (eval-data path,
target path, revision, tokenizer, template, aux layers, seqlen) is keyed —
flipping any one changes the key, so a target swap or template change can never
serve stale hidden states — while execution knobs (micro-batch size) are
deliberately NOT keyed, so re-batching hits the same entry. The batch-size gate
is the roadmap's regression proof that per-position aggregation happens over
the whole pass before the geometric sum: the same eval set at micro-batch sizes
1/4/16 yields the same ``simulated_acc_len``.

CPU except the batch-size gate (GPU; run on the H200 box via rcli).
"""

import os
import tempfile
import unittest

import torch

from specforge.eval import EvalCache, EvalConfig, Evaluator
from specforge.training.strategies.base import StepOutput

CUDA = torch.cuda.is_available()

BASE = dict(
    eval_data_path="/data/eval.jsonl",
    target_model_path="/models/llama",
    target_revision="abc123",
    tokenizer_path="/models/llama-tok",
    chat_template="llama3",
    aux_hidden_state_layer_ids=(2, 15, 28),
    max_len=2048,
)


class TestEvalCacheKey(unittest.TestCase):
    def test_every_identity_field_is_keyed(self):
        cache = EvalCache(tempfile.mkdtemp(prefix="evalcache_"))
        base_key = cache.key(EvalConfig(**BASE))
        # deterministic across instances
        self.assertEqual(base_key, EvalCache("/elsewhere").key(EvalConfig(**BASE)))
        flips = dict(
            eval_data_path="/data/eval2.jsonl",
            target_model_path="/models/qwen",
            target_revision="def456",
            tokenizer_path="/models/qwen-tok",
            chat_template="qwen",
            aux_hidden_state_layer_ids=(2, 15, 29),
            max_len=1024,
        )
        for field, value in flips.items():
            flipped = EvalConfig(**{**BASE, field: value})
            self.assertNotEqual(
                base_key, cache.key(flipped), f"{field} must be part of the key"
            )
        # Execution knobs are NOT identity: re-batching hits the same entry.
        self.assertEqual(
            base_key, cache.key(EvalConfig(**BASE, micro_batch_size=16))
        )

    def test_hit_produce_once_and_invalidate(self):
        cache = EvalCache(tempfile.mkdtemp(prefix="evalcache_"))
        cfg = EvalConfig(**BASE)
        produced = {"n": 0}

        def produce(staging):
            produced["n"] += 1
            torch.save({"x": torch.ones(2)}, os.path.join(staging, "000.ckpt"))

        first = cache.get_or_produce(cfg, produce)
        again = cache.get_or_produce(cfg, produce)
        self.assertEqual(first, again)
        self.assertEqual(produced["n"], 1)  # the point: no recompute on a hit
        self.assertTrue(os.path.exists(os.path.join(first, "000.ckpt")))
        self.assertTrue(os.path.exists(os.path.join(first, "eval_cache_meta.json")))
        # a different identity is a different entry
        other = EvalConfig(**{**BASE, "target_revision": "zzz"})
        self.assertIsNone(cache.try_get(other))
        # invalidate -> next call reproduces
        self.assertTrue(cache.invalidate(cfg))
        cache.get_or_produce(cfg, produce)
        self.assertEqual(produced["n"], 2)


def _stack_collate(features):
    keys = features[0].keys()
    return {k: torch.stack([f[k] for f in features], dim=0) for k in keys}


def _put_features(store, n=6, seq=8):
    """Identical deterministic features into any FeatureStore; returns refs."""
    g = torch.Generator().manual_seed(0)
    refs = []
    for i in range(n):
        tensors = {
            "input_ids": torch.randint(0, 100, (seq,), generator=g),
            "loss_mask": torch.ones(seq, dtype=torch.long),
        }
        refs.append(
            store.put(
                tensors,
                sample_id=f"s{i}",
                metadata={
                    "run_id": "eval",
                    "target_repr": "hidden_state",
                    "ttt_length": 2,
                },
            )
        )
    return refs


class TestEvalStreamEquivalence(unittest.TestCase):
    """Eval metrics must not depend on WHICH FeatureStore the stream reads from
    (colocated mem:// vs a disagg shared-dir store) — the loader erases it."""

    def _metrics_over(self, store, refs):
        from specforge.runtime.data_plane import FeatureDataLoader

        loader = FeatureDataLoader(
            store, refs=refs, batch_size=2, collate_fn=_stack_collate
        )

        def forward(batch):  # a deterministic pure function of the batch DATA
            loss = batch.tensors["input_ids"].float().mean()
            acc = batch.tensors["loss_mask"].float().mean() * 0.5
            return StepOutput(loss=loss, metrics={"accuracy": acc})

        return Evaluator().run(forward, loader)

    def test_local_vs_shared_dir_store(self):
        from specforge.runtime.data_plane import LocalFeatureStore
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore

        local = LocalFeatureStore("eval-local")
        m_local = self._metrics_over(local, _put_features(local))

        shared = SharedDirFeatureStore(
            tempfile.mkdtemp(prefix="evalshared_"), store_id="eval-shared"
        )
        m_shared = self._metrics_over(shared, _put_features(shared))

        self.assertEqual(m_local, m_shared)
        self.assertGreater(m_local["eval/avg_loss"], 0.0)


@unittest.skipUnless(CUDA, "loader-level batch-size gate requires CUDA")
class TestBatchSizeInvarianceOverLoader(unittest.TestCase):
    def test_simulated_acc_len_invariant_at_bs_1_4_16(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29572")

        from specforge.launch import _offline_io
        from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore
        from specforge.training.strategies.registry import resolve_strategy

        TTT, N, MAX_LEN = 3, 16, 512
        workdir = tempfile.mkdtemp(prefix="bs_inv_")
        # equal-length sequences: batching must not change padding, so any
        # metric difference would be an aggregation-ordering bug.
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=N, seq=64)

        torch.manual_seed(0)
        model, head = fx.build_eagle3(workdir, ttt=TTT)
        spec = resolve_strategy("eagle3")
        strategy = spec.make_strategy(model, target_head=head)
        strategy.trainable_module().eval()
        collate_fn, per_sample_transform = _offline_io(spec, MAX_LEN)
        refs = spec.make_offline_reader(
            feat_dir, run_id="bs", ttt_length=TTT, max_len=MAX_LEN
        ).read()
        store = LocalFeatureStore("bs")

        results = {}
        for bs in (1, 4, 16):
            loader = FeatureDataLoader(
                store,
                refs=refs,
                batch_size=bs,
                collate_fn=collate_fn,
                per_sample_transform=per_sample_transform,
            )
            results[bs] = Evaluator().run(
                lambda batch: strategy.forward_loss(batch), loader
            )

        for bs in (4, 16):
            self.assertAlmostEqual(
                results[1]["eval/simulated_acc_len"],
                results[bs]["eval/simulated_acc_len"],
                places=4,
                msg=f"simulated_acc_len moved between bs=1 and bs={bs}",
            )
            self.assertAlmostEqual(
                results[1]["eval/avg_loss"],
                results[bs]["eval/avg_loss"],
                places=4,
                msg=f"avg_loss moved between bs=1 and bs={bs}",
            )
            self.assertAlmostEqual(
                results[1]["eval/avg_acc"],
                results[bs]["eval/avg_acc"],
                places=4,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
