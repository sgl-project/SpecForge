# coding=utf-8
"""Training plane: trainer boundary split (controller / core / strategy / backend).

Submodules import the SpecForge model code, so they are imported explicitly by
callers rather than at package load (keeps the control/data plane importable
without a GPU/model environment).
"""
