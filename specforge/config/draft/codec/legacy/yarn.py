"""Resolve historical YaRN derivations into explicit computational values."""

import math

from ..._validation import finite_number, positive_int, positive_number


def expand_yarn(values, max_position_embeddings):
    values = dict(values)
    positive_int("max_position_embeddings", max_position_embeddings)
    original = values.setdefault(
        "original_max_position_embeddings", max_position_embeddings
    )
    positive_int("original_max_position_embeddings", original)
    if "factor" not in values:
        raise ValueError("legacy YaRN is missing factor")
    if values["factor"] is None:
        values["factor"] = max_position_embeddings / original
    factor = values["factor"]
    positive_number("factor", factor)
    mscale = values.pop("mscale", None)
    mscale_all = values.get("mscale_all_dim")
    for name, value in (("mscale", mscale), ("mscale_all_dim", mscale_all)):
        if value is not None:
            finite_number(name, value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    def scale(multiplier=1.0):
        return 1.0 if factor <= 1 else 0.1 * multiplier * math.log(factor) + 1.0

    if values.get("attention_factor") is None:
        values["attention_factor"] = (
            scale(mscale) / scale(mscale_all) if mscale and mscale_all else scale()
        )
    values["mscale_all_dim"] = 0.0 if mscale_all is None else mscale_all
    for name, default in (("beta_fast", 32.0), ("beta_slow", 1.0)):
        value = values.get(name)
        if value is not None:
            finite_number(name, value)
        # HF's historical `value or default` treats zero and null as unset.
        values[name] = default if value is None or value == 0 else value
    values.setdefault("truncate", True)
    return values
