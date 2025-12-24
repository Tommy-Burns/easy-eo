import importlib
import pkgutil
from pathlib import Path

def load_ops():
    """
    Auto-import all modules inside eeo/..
    This triggers @eeo_raster_op decorators.
    """
    for pkg_name in ("eeo.ops", "eeo.analysis", "eeo.preprocessing", "eeo.viz"):
        pkg = importlib.import_module(pkg_name)
        pkg_path = Path(pkg.__file__).parent

        for module in pkgutil.iter_modules([str(pkg_path)]):
            importlib.import_module(f"{pkg_name}.{module.name}")
