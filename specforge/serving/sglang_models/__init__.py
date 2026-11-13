# coding=utf-8
"""SGLang draft model classes for SpecForge drafters SGLang does not ship.

Point SGLang at this package and every module here that defines ``EntryClass``
registers its architectures (overriding same-named built-ins)::

    SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models \\
    python -m sglang.launch_server --model <target> \\
        --speculative-algorithm DSPARK --speculative-draft-model-path <export> ...

Modules:

- ``dflash_moe``: ``Qwen3MoeDSparkModel`` (plus ``DFlashMoEDraftModel`` /
  ``DFlash2MoEDraftModel``), the DFlash-family draft classes with the dense
  MLP replaced by SGLang's own MoE block (``TopK`` + ``FusedMoE`` + a
  sigmoid-gated shared expert), loading the export through SGLang's weight
  loaders and refusing partially matching checkpoints. The fused MoE kernel's
  tuned tile configs for the Qwen3.8-27B DSpark MoE drafter live in
  ``patches/sglang/moe_configs``.
- ``moe_ffn``: the same FFN in plain PyTorch (no SGLang import), the reference
  for ``scripts/gates/check_dspark_moe_sglang_equivalence.py`` and the tests.
"""
