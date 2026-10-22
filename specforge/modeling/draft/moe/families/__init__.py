# coding=utf-8
"""Target families: one module per family, imported here for registration.

A family module registers, from its own file only:

- its preset (:func:`~..config.register_moe_preset`): the routing recipe and
  the default checkpoint layout;
- its checkpoint layout (:func:`~..checkpoint.register_layout`): the family's
  file naming and the ``config.json`` fields its serving engine reads.

Adding a family adds its module and one import line below; nothing else in
the package changes. Core modules never import this package.
"""

from . import deepseek  # noqa: F401  (registers deepseek_v4 and its layout)
from . import qwen  # noqa: F401  (registers qwen3_5_moe and the qwen layout)
