# coding=utf-8
"""Data plane: large-tensor storage, transfer, and materialization.

Exports stay lazy so metadata-only control-plane users do not import PyTorch or
the Mooncake client merely by importing a sibling module.
"""

from importlib import import_module

__all__ = [
    "FeatureStore",
    "LocalFeatureStore",
    "load_feature_file",
    "spec_from_tensor",
    "SampleRefQueue",
    "LocalRolloutStream",
    "RefDistributor",
    "FeatureDataLoader",
    "OfflineManifestReader",
    "list_feature_files",
    "SharedDirFeatureStore",
    "MooncakeFeatureStore",
    "AuthPolicy",
]

_EXPORT_MODULE = {
    "FeatureStore": "feature_store",
    "LocalFeatureStore": "feature_store",
    "load_feature_file": "feature_store",
    "spec_from_tensor": "feature_store",
    "SampleRefQueue": "sample_ref_queue",
    "LocalRolloutStream": "local_rollout_stream",
    "RefDistributor": "ref_distributor",
    "FeatureDataLoader": "feature_dataloader",
    "OfflineManifestReader": "offline_reader",
    "list_feature_files": "offline_reader",
    "SharedDirFeatureStore": "disaggregated",
    "MooncakeFeatureStore": "mooncake_store",
    "AuthPolicy": "disaggregated",
}


def __getattr__(name):
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
