# coding=utf-8
"""PolicyFeatureAdapter batches target capture by sequence length."""

import unittest

import torch

from specforge.inference.adapters.policy import (
    EAGLE3_FEATURE_SCHEMA,
    PolicyFeatureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.inference.target_engine import Eagle3TargetOutput
from specforge.runtime.contracts import PromptTask

H, V = 4, 16


class FakeTarget:
    """Records each capture() call's batch size; encodes the first token id into
    hidden_states so per-sample slicing can be verified. Mirrors the TargetEngine
    contract consumed by the policy adapter."""

    capture_layers = [1, 2, 3]

    def __init__(self):
        self.call_batch_sizes = []

    def capture(self, input_ids, attention_mask, loss_mask, **kwargs):
        G, L = input_ids.shape
        self.call_batch_sizes.append(G)
        first_tok = input_ids[:, :1].float()  # (G,1)
        return Eagle3TargetOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask.unsqueeze(-1),
            hidden_states=first_tok.unsqueeze(-1).expand(G, L, 3 * H).clone(),
            target=torch.zeros(G, L, V),
        )


class TestAdapterBatching(unittest.TestCase):
    def _capture(self):
        return CaptureConfig.from_strategy(
            required_features={
                "input_ids",
                "attention_mask",
                "loss_mask",
                "hidden_state",
                "target",
            },
            aux_hidden_state_layer_ids=(1, 2, 3),
            target_repr="logits",
            target_hidden_size=H,
            target_vocab_size=V,
        )

    def test_groups_by_length_one_call_per_group(self):
        target = FakeTarget()
        adapter = PolicyFeatureAdapter(
            target, schema=EAGLE3_FEATURE_SCHEMA, device="cpu"
        )
        tasks = [
            PromptTask("t0", "r", "s", {"input_ids": [10, 11, 12, 13]}, 4),
            PromptTask("t1", "r", "s", {"input_ids": [20, 21, 22, 23]}, 4),
            PromptTask("t2", "r", "s", {"input_ids": [30, 31, 32, 33, 34, 35]}, 6),
        ]
        feats = adapter.generate_features(tasks, capture=self._capture())
        # 2 length-groups (len4 x2, len6 x1) -> 2 batched calls, not 3 per-sample
        self.assertEqual(sorted(target.call_batch_sizes), [1, 2])
        self.assertEqual(len(feats), 3)
        # order preserved + correct per-sample mapping (first-token encoding)
        self.assertEqual(feats[0]["hidden_state"].shape, (1, 4, 3 * H))
        self.assertEqual(feats[2]["hidden_state"].shape, (1, 6, 3 * H))
        self.assertEqual(int(feats[0]["hidden_state"][0, 0, 0]), 10)
        self.assertEqual(int(feats[1]["hidden_state"][0, 0, 0]), 20)
        self.assertEqual(int(feats[2]["hidden_state"][0, 0, 0]), 30)
        for f in feats:
            self.assertEqual(f["__aux_layer_ids__"], (1, 2, 3))
            self.assertEqual(f["target"].shape[-1], V)

    def test_rejects_unimplemented_target_repr(self):
        # the online adapter must only advertise reprs it implements
        adapter = PolicyFeatureAdapter(
            FakeTarget(), schema=EAGLE3_FEATURE_SCHEMA, device="cpu"
        )
        cap = CaptureConfig.from_strategy(
            required_features={
                "input_ids",
                "attention_mask",
                "loss_mask",
                "hidden_state",
                "target",
            },
            aux_hidden_state_layer_ids=(1, 2, 3),
            target_repr="hidden_state",  # not implemented online
            target_hidden_size=H,
            target_vocab_size=V,
        )
        task = PromptTask("t0", "r", "s", {"input_ids": [1, 2, 3, 4]}, 4)
        with self.assertRaises(NotImplementedError):
            adapter.generate_features([task], capture=cap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
