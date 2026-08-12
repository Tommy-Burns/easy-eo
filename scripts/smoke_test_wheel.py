#!/usr/bin/env python
"""Smoke-test an installed Easy-EO distribution.

The test suite runs against the source tree, so it cannot catch a packaging
mistake: a file the build backend did not include, a version that disagrees
with the tag, an entry point that only resolves because the working directory
happened to be the checkout. This script answers the separate question — does
the *built artifact* install, import, and do real work — and the release
workflow runs it on the wheel and the sdist before either is published, since
a PyPI release cannot be taken back.

It imports ``eeo`` the way a user would and refuses to run inside a source
checkout, because importing the working tree would prove nothing about the
artifact.

Usage
-----
    python scripts/smoke_test_wheel.py
    python scripts/smoke_test_wheel.py --expect-version 0.3.0
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
from affine import Affine

import eeo

# A north-up 1-unit grid. Without a real transform, promoting a dataset to
# rasterio warns about the identity matrix and clutters the release log.
TRANSFORM = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 8.0)

# Populated by the @check decorator, in definition order.
CHECKS: list[tuple[str, Callable[[argparse.Namespace], None]]] = []

# Files that ship as package data rather than as importable modules, and so are
# the ones a build-backend misconfiguration silently drops. py.typed makes the
# annotations visible to type checkers; core.pyi declares the dynamically bound
# operations; _registry.py pins the sample-data URLs and checksums.
PACKAGED_FILES = ("py.typed", "core/core.pyi", "datasets/_registry.py")

# A sample of the operations bound onto EEORasterDataset at import time by
# @eeo_raster_op / @eeo_raster_viz. Their absence means the decorator registry
# did not run, which an artifact can break without any test noticing.
BOUND_OPERATIONS = (
    "add",
    "multiply",
    "ndvi",
    "normalize_min_max",
    "clip_raster_with_vector",
    "resample",
    "reproject_raster",
    "stack",
    "mosaic",
    "get_mean_pixel",
    "plot_raster",
)


def check(
    description: str,
) -> Callable[[Callable[[argparse.Namespace], None]], Callable[[argparse.Namespace], None]]:
    """Register a check to run.

    Parameters
    ----------
    description : str
        One-line summary, printed with the check's pass/fail result.

    Returns
    -------
    callable
        Decorator that registers the function and returns it unchanged.
    """

    def decorator(
        func: Callable[[argparse.Namespace], None],
    ) -> Callable[[argparse.Namespace], None]:
        CHECKS.append((description, func))
        return func

    return decorator


@check("eeo imports from an installed distribution, not a source checkout")
def check_installed(args: argparse.Namespace) -> None:
    """Verify the import resolved to an installed package.

    Every later check is meaningless if this one fails: they would all be
    exercising the working tree.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (unused).
    """
    package_dir = Path(eeo.__file__).resolve().parent
    if not any(part in ("site-packages", "dist-packages") for part in package_dir.parts):
        raise AssertionError(
            f"eeo imported from {package_dir}, which is not an installed "
            f"location. Install the built artifact and run this script from "
            f"outside the source checkout."
        )


@check("__version__, package metadata, and the release tag agree")
def check_version(args: argparse.Namespace) -> None:
    """Verify the version is consistent, and matches the tag when given.

    A tag that disagrees with ``__version__`` publishes a file whose name
    promises one version and whose contents report another.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments; ``expect_version`` is compared against
        the installed version when it is not None.
    """
    dist_version = importlib.metadata.version("easy-eo")
    if eeo.__version__ != dist_version:
        raise AssertionError(
            f"eeo.__version__ is {eeo.__version__!r} but the installed "
            f"distribution reports {dist_version!r}"
        )
    if args.expect_version is not None and dist_version != args.expect_version:
        raise AssertionError(
            f"installed version is {dist_version!r} but the tag expects {args.expect_version!r}"
        )


@check("package data files are present")
def check_packaged_files(args: argparse.Namespace) -> None:
    """Verify the non-module files listed in PACKAGED_FILES shipped.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (unused).
    """
    package_dir = Path(eeo.__file__).resolve().parent
    missing = [name for name in PACKAGED_FILES if not (package_dir / name).is_file()]
    if missing:
        raise AssertionError(f"missing from the artifact: {', '.join(missing)}")


@check("every name in __all__ is importable")
def check_public_api(args: argparse.Namespace) -> None:
    """Verify the documented public surface resolves.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (unused).
    """
    missing = [name for name in eeo.__all__ if not hasattr(eeo, name)]
    if missing:
        raise AssertionError(f"exported but not importable: {', '.join(missing)}")


@check("chainable operations are bound onto EEORasterDataset")
def check_bound_operations(args: argparse.Namespace) -> None:
    """Verify the decorator registry bound its operations.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (unused).
    """
    dataset_class = type(eeo.load_array(np.zeros((2, 2), dtype="float32")))
    missing = [name for name in BOUND_OPERATIONS if not hasattr(dataset_class, name)]
    if missing:
        raise AssertionError(f"not bound onto EEORasterDataset: {', '.join(missing)}")


@check("a real chain computes correct values and preserves metadata")
def check_ndvi_chain(args: argparse.Namespace) -> None:
    """Run a load → index → read chain and verify the result.

    Uses uniform bands so the expected NDVI is exact:
    ``(0.6 - 0.2) / (0.6 + 0.2) == 0.5``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (unused).
    """
    scene = np.stack([np.full((8, 8), 0.6, dtype="float32"), np.full((8, 8), 0.2, dtype="float32")])
    dataset = eeo.load_array(scene, crs=4326, transform=TRANSFORM, band_names=["nir", "red"])
    result = dataset.ndvi(red="red", nir="nir", name="ndvi")
    values = result.read()

    if values.dtype != np.dtype("float32"):
        raise AssertionError(f"expected float32 output, got {values.dtype}")
    if not np.allclose(values, 0.5):
        raise AssertionError(f"expected NDVI 0.5 everywhere, got {np.unique(values)}")
    if values.shape[-2:] != (8, 8):
        raise AssertionError(f"expected an 8x8 result, got shape {values.shape}")
    if result.get_crs().to_epsg() != 4326:
        raise AssertionError(f"CRS not preserved through the chain: {result.get_crs()}")


@check("a missing optional extra raises MissingDependencyError with an install hint")
def check_missing_extra(args: argparse.Namespace) -> None:
    """Verify the absent-extra path reports the documented error.

    Skipped when rioxarray is installed, since the check is about what happens
    when it is not.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (unused).
    """
    if importlib.util.find_spec("rioxarray") is not None:
        return

    dataset = eeo.load_array(np.zeros((4, 4), dtype="float32"), crs=4326, transform=TRANSFORM)
    try:
        dataset.to_xarray()
    except eeo.MissingDependencyError as err:
        if "install" not in str(err).lower():
            raise AssertionError(f"error names no install command: {err}") from err
    else:
        raise AssertionError("to_xarray() succeeded without rioxarray installed")


@check("show_versions() runs")
def check_show_versions(args: argparse.Namespace) -> None:
    """Verify the diagnostic helper the bug-report template points at works.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (unused).
    """
    eeo.show_versions()


def main() -> int:
    """Run every registered check and report the results.

    Returns
    -------
    int
        0 if every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-version",
        default=None,
        help="version the artifact must report, e.g. the release tag without its 'v'",
    )
    args = parser.parse_args()

    print(f"Smoke-testing easy-eo from {Path(eeo.__file__).resolve().parent}\n")

    failures = 0
    for description, func in CHECKS:
        try:
            func(args)
        except Exception:
            failures += 1
            print(f"FAIL  {description}")
            traceback.print_exc()
            print()
        else:
            print(f"ok    {description}")

    print()
    if failures:
        print(f"{failures} of {len(CHECKS)} checks failed; the artifact is not releasable.")
        return 1
    print(f"All {len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
