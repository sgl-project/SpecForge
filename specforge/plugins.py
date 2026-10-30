"""Out-of-tree platform plugins.

A plugin is a zero-argument callable published under the ``specforge.plugins``
entry-point group. It runs once per process, before any training module is
imported, and decides by itself whether its platform is active.
"""

from __future__ import annotations

from importlib.metadata import entry_points

PLUGIN_GROUP = "specforge.plugins"

_loaded = False


def load_plugins() -> None:
    """Call every installed plugin once per process."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for entry_point in entry_points(group=PLUGIN_GROUP):
        entry_point.load()()
