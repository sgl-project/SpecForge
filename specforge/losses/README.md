# Shared numerical losses

This package contains numerical functions used by both canonical objectives and
legacy training. It does not register algorithms or choose execution policies.

| Module | Purpose |
| --- | --- |
| `fused.py` | Reference and fused token losses |
| `domino.py` | Weighted loss for base logits and suffix corrections |
| `lk.py` | Acceptance metrics and LK losses |
| `position_weights.py` | Detached position weighting |
| `soft_cross_entropy.py` | Historical soft cross-entropy autograd implementation |

Private Triton modules implement the kernels. Importing portable loss modules
does not require Triton; `soft_cross_entropy` retains its existing Triton dependency.
The package initializer does not eagerly import these modules.

`algorithms/objectives/implementations/` adapts numerical functions to execution
requests and selects optimizations. Chunk scheduling and teacher preparation are
outside this package.

The previous loss paths under `specforge.core` and
`algorithms.objectives.implementations.fused_loss` have been removed. Import
numerical functions from this package. Numerical formulas, gradients, defaults
and reduction behavior are unchanged.
