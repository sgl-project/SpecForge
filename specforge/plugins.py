"""Platform plugins loaded when SpecForge's CLI is imported.

A plugin publishes a zero-argument callable in the ``specforge.plugins``
entry-point group and decides whether its platform is active.
"""

from functools import cache
from importlib.metadata import EntryPoint, entry_points

PLUGIN_GROUP = "specforge.plugins"


@cache
def _activate(entry_point: EntryPoint) -> None:
    """Remember successful activations so a retry skips earlier plugins."""
    try:
        entry_point.load()()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to activate SpecForge plugin {entry_point.name!r}"
        ) from exc


@cache
def load_plugins() -> None:
    """Activate the installed CLI plugins before the CLI imports configuration."""
    for entry_point in entry_points(group=PLUGIN_GROUP):
        _activate(entry_point)
