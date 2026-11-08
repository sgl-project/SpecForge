"""One-way conversions to the explicitly pinned SGLang checkpoint format."""

SGLANG_COMMIT = "c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0"
CONVERTER_VERSION = 1
# HF container/dtype probes do not select draft computation in this format.
NON_COMPUTATIONAL_READS = frozenset(
    {"text_config", "decoder", "generator", "text_encoder", "torch_dtype", "dtype"}
)
_CONVERTERS = {}


def register_converter(architecture):
    def register(convert):
        if architecture in _CONVERTERS:
            raise ValueError(f"duplicate SGLang converter: {architecture}")
        _CONVERTERS[architecture] = convert
        return convert

    return register


def convert_config(config):
    try:
        converter = _CONVERTERS[config.architecture]
    except KeyError:
        raise ValueError(f"no SGLang converter for {config.architecture}") from None
    return converter(config)


def artifact_metadata():
    return dict(
        format="sglang",
        version=1,
        converter_version=CONVERTER_VERSION,
        sglang_commit=SGLANG_COMMIT,
    )


# Import for registration, after the registry is defined.
from . import converters  # noqa: E402,F401
