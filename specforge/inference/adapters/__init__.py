# coding=utf-8
"""FeatureSource adapters: schema-parameterized capture over a TargetEngine.

``policy.PolicyFeatureAdapter`` is the single runtime adapter (length grouping,
batched capture, per-sample slicing, vocab projection); each strategy registers
a ``FeatureSchema`` for the store-ready dict shape. ``eagle3.SGLangAdapter``
and ``dflash.DFlashAdapter`` are thin schema-pinning subclasses kept for
back-compat; all implement the ``rollout_worker.FeatureSource`` protocol.

``server_capture.SGLangServerCaptureAdapter`` is the zero-copy SERVER
transport (``rollout_worker.RefSource``): a patched live SGLang server writes
the features straight into the Mooncake store and the adapter returns
committed-ready ``SampleRef``s — import it from its module (it stays out of
this package ``__init__`` so the registry import path remains light).
"""
