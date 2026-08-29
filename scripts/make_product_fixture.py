#!/usr/bin/env python
"""Materialise one of the suite's synthetic products into a real directory.

Maintainer tool, not part of the shipped library. The test suite builds these
products into a temporary directory and throws them away, which is right for
tests and unhelpful the moment you want to open it in a GIS software.

This writes the same products, from the same builders in
``tests/product_fixtures.py``, somewhere they persist. There is no second
definition to drift: change the fixture module and both the suite and this
script follow.

Usage
-----
    python scripts/make_product_fixture.py sentinel2 --out build/fixtures
    python scripts/make_product_fixture.py landsat --mission 7 --out build/fixtures
    python scripts/make_product_fixture.py sentinel2 --out build/fixtures --archive

``--archive`` additionally packs the product the way it would have been
downloaded: a ``.zip`` for Sentinel-2, a ``.tar`` for Landsat.

Requires only rasterio + numpy (already runtime deps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The builders live with the tests, because that is what they exist for.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from product_fixtures import (  # noqa: E402
    L7_ID,
    L9_ID,
    build_landsat,
    build_safe,
    pack_tar,
    pack_zip,
)

#: Landsat missions with a band table, and the product id to build for each.
_LANDSAT_IDS = {7: L7_ID, 9: L9_ID}


def _sentinel2(out: Path, *, archive: bool) -> tuple[Path, Path | None]:
    """Build a Level-2A .SAFE product, optionally zipped as Copernicus ships it."""
    safe = build_safe(out)
    if not archive:
        return safe, None
    return safe, pack_zip(safe, out / f"{safe.name[:-5]}.zip", base=out)


def _landsat(out: Path, *, mission: int, archive: bool) -> tuple[Path, Path | None]:
    """Build a Collection 2 Level-2 product, optionally tarred as USGS ships it."""
    product = build_landsat(out, product_id=_LANDSAT_IDS[mission])
    if not archive:
        return product, None
    # USGS tars a product's files at the root, with no enclosing directory.
    return product, pack_tar(product, out / f"{product.name}.tar", base=product)


def main(argv: list[str] | None = None) -> int:
    """Build one product and report where it landed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mission", choices=["sentinel2", "landsat"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/fixtures"),
        help="directory to build into (default: build/fixtures)",
    )
    parser.add_argument(
        "--landsat-mission",
        dest="landsat_mission",
        type=int,
        choices=sorted(_LANDSAT_IDS),
        default=9,
        help="which Landsat to build; 7 uses the TM/ETM+ band table (default: 9)",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="also pack the product as it would have been downloaded",
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.mission == "sentinel2":
        built, packed = _sentinel2(args.out, archive=args.archive)
    else:
        built, packed = _landsat(args.out, mission=args.landsat_mission, archive=args.archive)

    written = [path for path in built.rglob("*") if path.is_file()]
    size = sum(path.stat().st_size for path in written)
    print(f"built {built} ({len(written)} files, {size / 1e3:.0f} kB)")
    if packed is not None:
        print(f"packed {packed} ({packed.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
