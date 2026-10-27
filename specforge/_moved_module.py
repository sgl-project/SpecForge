"""Keep relocated modules importable from their previous paths for one release."""

import importlib
import sys
import warnings


def alias_moved_module(old: str, new: str) -> None:
    """Make ``old`` resolve to the module object at ``new``.

    Replacing the ``sys.modules`` entry while ``old`` is being imported makes the
    import return ``new`` itself, so classes, registries and patch targets stay
    single objects instead of being duplicated under two names.
    """

    warnings.warn(
        f"{old} has moved to {new}; the old import path will be removed in a "
        "future release",
        DeprecationWarning,
        stacklevel=3,
    )
    sys.modules[old] = importlib.import_module(new)
