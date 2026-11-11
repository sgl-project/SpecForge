"""Independent references for the Markov gates reused by legacy DSpark."""

import unittest

import torch
import torch.nn.functional as F

from specforge.modeling.draft import dspark
from specforge.modeling.draft.heads.markov import GatedMarkovHead, RNNMarkovHead


class MarkovHeadTest(unittest.TestCase):
    def test_gated_and_recurrent_equations(self):
        torch.manual_seed(0)
        hidden, logits = torch.randn(2, 3, 8), torch.randn(2, 3, 11)
        ids = torch.randint(0, 11, (2, 3))
        for kind in (GatedMarkovHead, RNNMarkovHead):
            head = kind(vocab_size=11, markov_rank=4, hidden_size=8)
            self.assertIs(getattr(dspark, kind.__name__), kind)
            embeddings = F.embedding(ids, head.markov_w1.weight)
            if kind is GatedMarkovHead:
                gate = F.linear(
                    torch.cat((hidden, embeddings), -1),
                    head.gate_proj.weight,
                    head.gate_proj.bias,
                ).sigmoid()
                bias = F.linear(gate * embeddings, head.markov_w2.weight)
            else:
                state, biases = torch.zeros(2, 4), []
                for index in range(3):
                    values = F.linear(
                        torch.cat((state, embeddings[:, index], hidden[:, index]), -1),
                        head.joint_proj.weight,
                        head.joint_proj.bias,
                    )
                    gate, candidate, output = values.chunk(3, -1)
                    state = (
                        gate.sigmoid() * state + (1 - gate.sigmoid()) * candidate.tanh()
                    )
                    biases.append(F.linear(output.tanh(), head.markov_w2.weight))
                bias = torch.stack(biases, 1)
            actual = head.apply_block_logits(
                logits, token_ids=ids, hidden_states=hidden
            )
            torch.testing.assert_close(actual, logits + bias, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
