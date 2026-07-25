"""Dataset adapters: raw vendor columns -> this repo's canonical schema.

Each adapter exposes the same three names so ``src/training/tabular.py`` can load
any problem without branching:

``CANONICAL_COLUMNS``
    Ordered ``{column: dtype}`` describing what ``load()`` guarantees to return.

``PROVENANCE``
    Where the data came from, under what licence, and what it costs to obtain.

``load(path=None, *, sample=False)``
    Returns a ``pandas.DataFrame`` in the canonical schema. With ``sample=True``
    it reads the committed ``data/sample/*.csv`` fixture instead of the real
    dataset, so CI never needs credentials.

An adapter never invents a label. If the source has no usable outcome column,
the adapter raises rather than deriving one.
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Adapter module paths, keyed by problem name.
ADAPTERS = {
    "fraud": "src.data.adapters.ieee_cis",
    "credit_risk": "src.data.adapters.lending_club",
    "churn": "src.data.adapters.credit_card_churn",
}

_REQUIRED_ATTRS = ("CANONICAL_COLUMNS", "PROVENANCE", "load")


def get_adapter(dotted_path: str) -> ModuleType:
    """Import an adapter module by dotted path and check it satisfies the contract.

    Args:
        dotted_path: e.g. ``"src.data.adapters.ieee_cis"``.

    Returns:
        The imported module.

    Raises:
        KeyError: The module imports but does not satisfy the adapter contract.
    """
    module = importlib.import_module(dotted_path)
    missing = [name for name in _REQUIRED_ATTRS if not hasattr(module, name)]
    if missing:
        raise KeyError(
            f"{dotted_path} is not a valid adapter: missing {missing}. "
            f"Known adapters: {sorted(ADAPTERS.values())}"
        )
    return module
