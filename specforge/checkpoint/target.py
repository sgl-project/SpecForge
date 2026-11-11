"""Shared target table resolution for draft bindings and teacher projection."""

from .tensors import TensorStore
from .weight_plan import _validate_unquantized

_TARGET_KEYS = {
    "embed_tokens": (
        "language_model.model.embed_tokens.weight",
        "embed.weight",
        "model.llm.embed.weight",
        "model.word_embeddings.weight",
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ),
    "lm_head": (
        "language_model.lm_head.weight",
        "head.weight",
        "model.llm.unembed.weight",
        "model.language_model.lm_head.weight",
        "model.lm_head.weight",
        "lm_head.weight",
    ),
}


class TargetWeights(TensorStore):
    def __init__(self, source, *, cache_dir=None, keys=None):
        from .resolve import resolve_directory, target_configuration

        super().__init__(resolve_directory(source, cache_dir=cache_dir))
        self.config = target_configuration(self.root)
        self.keys = keys or {}
        self._tables = {}

    def read(self, key, region=None):
        if region is not None:
            return super().read(key, region)
        if key not in self._tables:
            self._tables[key] = super().read(key)
        return self._tables[key]

    def resolve(self, name):
        return resolve_target_tensor(self, self.config, name)


def resolve_target_tensor(store, config, name):
    refs = store.refs()
    overrides = store.keys if isinstance(store, TargetWeights) else {}

    def candidates(module):
        keys = (overrides[module],) if module in overrides else _TARGET_KEYS[module]
        return [key for key in keys if key in refs]

    found = candidates(name)
    if (
        name == "lm_head"
        and not found
        and name not in overrides
        and config.get("tie_word_embeddings") is True
    ):
        found = candidates("embed_tokens")
    if len(found) != 1:
        raise ValueError(f"{name}: expected one target key, found {found}")
    ref = refs[found[0]]
    _validate_unquantized(name, ref)
    return ref
