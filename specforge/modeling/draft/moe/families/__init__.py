# coding=utf-8
"""Target families: one module per family, imported here for registration.

A family module registers, from its own file only:

- its preset (:func:`~..config.register_moe_preset`): the routing recipe and
  the default checkpoint layout;
- its checkpoint layout (:func:`~..checkpoint.register_layout`): the family's
  file naming.

Adding a family adds its module and one import line below; nothing else in
the package changes. Core modules never import this package.
"""
