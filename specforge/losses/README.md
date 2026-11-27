# Shared numerical losses

This package contains numerical functions used by canonical objectives. It does
not register algorithms or choose execution policies.

| Module | Purpose |
| --- | --- |
| `fused.py` | Reference and fused token losses |
| `domino.py` | Weighted loss for base logits and suffix corrections |
| `position_weights.py` | Detached position weighting |

Private Triton modules implement the kernels. Importing loss modules does not
require Triton. The package initializer does not eagerly import these modules.

`algorithms/objectives/implementations/` adapts numerical functions to execution
requests and selects optimizations. Chunk scheduling and teacher preparation are
outside this package.

The previous loss paths under `specforge.core` and
`algorithms.objectives.implementations.fused_loss` have been removed. Import
numerical functions from this package. Numerical formulas, gradients, defaults
and reduction behavior are unchanged.
