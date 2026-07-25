"""Import helper for the packages behind Easy-EO's optional extras.

Heavier capabilities — STAC data access, xarray interop, the lazy backend —
live behind optional extras so that a plain ``pip install easy-eo`` stays
small. The modules implementing them import their third-party packages through
:func:`import_optional`, which turns a missing package into a
:class:`~eeo.MissingDependencyError` naming the exact install command instead
of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from .core.exceptions import MissingDependencyError


def import_optional(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import a module provided by an optional extra.

    Parameters
    ----------
    module : str
        Importable module name (e.g. ``"pystac_client"``), not the
        distribution name.
    extra : str
        Name of the Easy-EO extra that provides the module (e.g. ``"stac"``),
        used to build the ``pip install`` hint.
    purpose : str
        Short description of the feature needing the module (e.g. ``"STAC
        search"``), used to open the error message.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    MissingDependencyError
        If the module is not installed. The message names the feature, the
        missing package, and the ``pip install 'easy-eo[<extra>]'`` command
        that provides it.

    Examples
    --------
    >>> from eeo._optional import import_optional
    >>> np = import_optional("numpy", extra="stac", purpose="STAC search")
    >>> np.__name__
    'numpy'
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as err:
        # A ModuleNotFoundError raised *inside* the optional package (one of
        # its own dependencies missing) is a different problem; report it as-is
        # rather than blaming the package the user did install.
        if err.name is not None and err.name.split(".")[0] != module.split(".")[0]:
            raise
        raise MissingDependencyError(
            f"{purpose} requires the optional '{module}' package, which is not "
            f"installed. Install the '{extra}' extra with: "
            f"pip install 'easy-eo[{extra}]'"
        ) from err
