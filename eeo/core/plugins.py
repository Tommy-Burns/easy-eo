"""Plugin registration: auto-import op modules so decorators bind methods."""

import importlib
import pkgutil
from pathlib import Path


def load_ops():
    """Auto-import all op modules so their decorators bind methods.

    Imports every module under ``eeo.ops``, ``eeo.analysis``,
    ``eeo.preprocessing``, and ``eeo.viz``, triggering the ``@eeo_raster_op``
    and ``@eeo_raster_viz`` decorators that bind methods onto
    ``EEORasterDataset``.
    """
    for pkg_name in ("eeo.ops", "eeo.analysis", "eeo.preprocessing", "eeo.viz"):
        pkg = importlib.import_module(pkg_name)
        if pkg.__file__ is None:  # namespace package without a concrete path
            continue
        pkg_path = Path(pkg.__file__).parent

        for module in pkgutil.iter_modules([str(pkg_path)]):
            importlib.import_module(f"{pkg_name}.{module.name}")
