"""Canonical model construction, independent of formats and training state."""

from inspect import isabstract

from specforge.config.draft import DraftConfig
from specforge.config.draft.registry import architecture_spec

from ..hf_config import SpecForgeDraftConfig
from ..pretrained import DraftPreTrainedModel

_MODELS: dict[str, type[DraftPreTrainedModel]] = {}


def register_draft_model(cls: type[DraftPreTrainedModel]) -> type[DraftPreTrainedModel]:
    """Register a concrete model under its class name, matching its schema name.

    Configuration schemas register separately in config.draft. No configuration
    alias, training objective, checkpoint adapter or loader is registered here.
    """
    if not isinstance(cls, type) or not issubclass(cls, DraftPreTrainedModel):
        raise TypeError("canonical models must inherit DraftPreTrainedModel")
    if isabstract(cls):
        raise TypeError("canonical model implementations must be concrete")
    architecture_spec(cls.__name__)
    if cls.config_class is not SpecForgeDraftConfig:
        raise TypeError("canonical models must use SpecForgeDraftConfig")
    if cls._supports_attention_backend is not False:
        raise ValueError(
            "draft models must disable the generic Transformers serving backend"
        )
    existing = _MODELS.get(cls.__name__)
    if existing is not None and existing is not cls:
        raise ValueError(f"model {cls.__name__!r} is already registered")
    _MODELS[cls.__name__] = cls
    return cls


def resolve_draft_model(architecture: str) -> type[DraftPreTrainedModel]:
    """Resolve a registered implementation; never fall back to legacy models."""
    architecture_spec(architecture)
    try:
        return _MODELS[architecture]
    except KeyError:
        raise ValueError(
            f"no canonical model registered for {architecture!r}"
        ) from None


def build_draft_model(config: DraftConfig, **model_kwargs) -> DraftPreTrainedModel:
    """Construct from the already resolved golden configuration, without I/O.

    Runtime kwargs (such as device, dtype or backend) go to the model constructor.
    This entry point neither reads files nor restores weights or training state.
    """
    if type(config) is not DraftConfig:
        raise TypeError("build_draft_model requires a resolved DraftConfig")
    model_class = resolve_draft_model(config.architecture)
    model = model_class(SpecForgeDraftConfig(config), **model_kwargs)
    if (
        not isinstance(model.config, SpecForgeDraftConfig)
        or model.config.draft is not config
    ):
        raise ValueError("model must retain the supplied golden DraftConfig")
    return model
