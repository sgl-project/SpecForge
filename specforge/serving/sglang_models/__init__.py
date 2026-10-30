# coding=utf-8
"""SGLang draft model classes for SpecForge drafters SGLang does not ship.

Point SGLang at this package; every module here that defines ``EntryClass``
registers its architectures (overriding same-named built-ins)::

    SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models \\
    python -m sglang.launch_server --model-path <target> \\
        --speculative-algorithm DSPARK --speculative-draft-model-path <export> ...

``dflash_moe`` serves the ``qwen3_5_moe`` drafts: ``Qwen3MoeDSparkModel`` and
``DFlashMoEDraftModel`` replace each layer's dense MLP with SGLang's own MoE
block (``TopK`` + ``FusedMoE`` + an optional sigmoid-gated shared expert).
"""
