# coding=utf-8
"""Serving-side pieces that run inside an inference engine, not the trainer.

``specforge.serving.sglang_models`` holds SGLang draft model classes for
drafters SGLang itself cannot load yet (the MoE-FFN DFlash family). SGLang
picks them up through its ``SGLANG_EXTERNAL_MODEL_PACKAGE`` hook, so the
installed SGLang stays unpatched.
"""
