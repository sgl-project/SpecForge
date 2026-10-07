"""Keep eager BF16 rounding boundaries when compiling native training."""

from torch._inductor import config as inductor_config


def compiler_numerics():
    # This scope must cover tracing, not just the final Inductor pass: make_fx
    # records low-precision barriers when its dispatch mode is constructed.
    # Both native block compilation and GraphTrainer use the same policy.
    # Include division_rounding so the upstream graph pass's global assignment
    # is also restored when this scope exits.
    return inductor_config.patch(
        {
            "emulate_precision_casts": True,
            "eager_numerics.division_rounding": True,
        }
    )
