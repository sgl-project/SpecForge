"""Draft components and composed models, with lazy historical API compatibility.

New callers use modeling.build_draft_model and loading.load_draft. Historical
training still has backend/parallelism features absent from the canonical path.
Import those implementations only when the old API is explicitly used.
"""

__all__ = [
    "Eagle3DraftModel",
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "DominoDraftModel",
    "DSparkDraftModel",
    "DSpineDraftModel",
    "LlamaForCausalLMEagle3",
    "PEagleDraftModel",
    "build_target_layer_ids",
    "extract_context_feature",
    "sample",
    "DRAFT_REGISTRY",
    "register_draft",
    "resolve_draft",
    "available_drafts",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import legacy

    value = getattr(legacy, name)
    globals()[name] = value
    return value
