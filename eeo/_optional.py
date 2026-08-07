"""Import helper for the packages behind Easy-EO's optional extras.

Heavier capabilities — STAC data access, xarray interop, the lazy backend —
live behind optional extras so that a plain ``pip install easy-eo`` stays
small. The modules implementing them import their third-party packages through
:func:`import_optional`, which turns a missing package into a
:class:`~eeo.MissingDependencyError` naming an install command that works for
the package manager Easy-EO was actually installed with, instead of a bare
``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from .core.exceptions import MissingDependencyError

# conda-forge package names for each extra's dependencies.
#
# conda has no concept of extras: a conda package carries one flat dependency
# list with nothing to opt into, and `conda install "easy-eo[stac]"` does not
# merely miss the extra — it fails to parse, because brackets already mean
# key-value constraints in conda's match syntax. So a conda user installs the
# same packages by name. The names happen to equal the PyPI ones for every
# current extra, but that is not a rule (conda-forge ships Matplotlib as
# `matplotlib-base`), which is why the mapping is written out rather than
# derived. tests/test_optional_dependencies.py checks it covers every declared
# extra, so a new extra cannot ship without its conda equivalent.
_CONDA_PACKAGES: dict[str, tuple[str, ...]] = {
    "stac": ("pystac-client", "planetary-computer"),
    "xarray": ("xarray", "rioxarray"),
}


def _installed_by_conda() -> bool | None:
    """Report whether conda manages this Easy-EO install.

    What matters is how *Easy-EO itself* arrived, not how the environment was
    created: pip-installed into a conda environment still calls for the pip
    instruction. conda's own record in ``conda-meta/`` is therefore the
    authority. The ``INSTALLER`` file in ``.dist-info`` cannot serve here — the
    conda-forge recipe builds with ``python -m pip install .``, so it reads
    ``pip`` for both install routes.

    Returns
    -------
    bool or None
        ``True`` when conda manages the install, ``False`` when it does not,
        and ``None`` when the environment could not be inspected — in which
        case the caller offers both commands rather than guessing.
    """
    try:
        conda_meta = Path(sys.prefix) / "conda-meta"
        if not conda_meta.is_dir():
            return False
        # Records are named "<name>-<version>-<build>.json", so the glob alone
        # would also match a hypothetical "easy-eo-stac" metapackage.
        return any(
            record.stem.rsplit("-", 2)[0] == "easy-eo"
            for record in conda_meta.glob("easy-eo-*.json")
        )
    except OSError:
        return None


def _install_hint(extra: str) -> str:
    """Build the install instruction for a missing extra.

    Parameters
    ----------
    extra : str
        Name of the Easy-EO extra that provides the missing package.

    Returns
    -------
    str
        A sentence naming the command to run, matching the package manager
        Easy-EO was installed with where that can be determined.
    """
    pip_command = f"pip install 'easy-eo[{extra}]'"
    pip_hint = f"Install the '{extra}' extra with: {pip_command}"

    packages = _CONDA_PACKAGES.get(extra)
    if packages is None:
        return pip_hint

    conda_command = f"conda install -c conda-forge {' '.join(packages)}"
    by_conda = _installed_by_conda()
    if by_conda is True:
        return (
            f"Easy-EO was installed by conda, which has no equivalent of pip's "
            f"extras, so install the '{extra}' extra's packages by name: "
            f"{conda_command}. Do not pip install them into a conda-managed "
            f"environment: conda does not track pip-installed files, so a later "
            f"conda install or update can overwrite them."
        )
    if by_conda is False:
        return pip_hint
    return f"{pip_hint}; or, if Easy-EO came from conda: {conda_command}"


def import_optional(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import a module provided by an optional extra.

    Parameters
    ----------
    module : str
        Importable module name (e.g. ``"pystac_client"``), not the
        distribution name.
    extra : str
        Name of the Easy-EO extra that provides the module (e.g. ``"stac"``),
        used to build the install hint.
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
        missing package, and the command that provides it — ``pip install
        'easy-eo[<extra>]'`` for a pip install, or the equivalent
        ``conda install -c conda-forge ...`` when conda manages Easy-EO, since
        conda has no extras mechanism.

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
            f"installed. {_install_hint(extra)}"
        ) from err
