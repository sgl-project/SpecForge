"""Draft and target modeling implementations.

Import concrete types from their owning modules; this package does not maintain
the pre-runtime aggregate model API. Canonical construction accepts a resolved
DraftConfig and leaves files, weights, and training state to their owners.
"""


def build_draft_model(config, **model_kwargs):
    """Build a registered canonical draft without eagerly importing model code."""
    from .draft.models import build_draft_model as build

    return build(config, **model_kwargs)


__all__ = ["build_draft_model"]
