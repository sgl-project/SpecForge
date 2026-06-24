# coding=utf-8
"""Training plane: trainer boundary split (controller / core / strategy / backend).

Submodules import ``torch`` (and, in the backend, ``specforge.distributed`` /
``yunchang`` lazily) and operate on built SpecForge model objects passed in by
the caller, so they are imported explicitly by training entry points rather than
at package load (keeps the control/data plane importable without a GPU/model
environment).
"""
