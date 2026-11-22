"""Ring communication and optional CUDA FlashAttention entry points."""

__all__ = [
    "ring_flash_attn_func",
    "ring_flash_attn_kvpacked_func",
    "ring_flash_attn_qkvpacked_func",
]


def __getattr__(name):
    if name in __all__:
        from . import ring_flash_attn

        return getattr(ring_flash_attn, name)
    raise AttributeError(name)
