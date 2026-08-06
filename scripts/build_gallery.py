"""Regenerate the README gallery images from the hosted sample dataset.

Maintainer tool, not part of the package. Every image is produced by an
Easy-EO plotting call on the public sample data, so the gallery shows what
the library actually renders with its defaults - no hand-tuned figures.

The captions in ``README.md`` quote the call that produced each image; if you
change a call here, change the caption there too.

Usage--
    python scripts/build_gallery.py

Writes PNGs to ``.github/assets/gallery/``. Needs network on the first run
(the sample data is downloaded and cached); the ``[xarray]``/``[stac]`` extras
are not required.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

from PIL import Image  # noqa: E402

import eeo  # noqa: E402
from eeo.datasets import load_sample_dataset  # noqa: E402

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / ".github" / "assets" / "gallery"

# Figure size in inches
FIGSIZE = (5, 5)

# Maximum width in the committed images
MAX_WIDTH = 800


def _shrink(path: pathlib.Path) -> None:
    """Downscale a rendered figure to web size, in place."""
    before = path.stat().st_size
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.width > MAX_WIDTH:
            height = round(img.height * MAX_WIDTH / img.width)
            img = img.resize((MAX_WIDTH, height), Image.LANCZOS)
        if path.suffix == ".jpg":
            img.save(path, format="JPEG", quality=90, optimize=True)
        else:
            img.save(path, format="PNG", optimize=True)
    after = path.stat().st_size
    print(f"  {path.name:<28} {before / 1e6:5.2f} MB -> {after / 1e6:5.2f} MB  {img.size}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sd = load_sample_dataset()
    scene = eeo.load_raster(sd.sentinel2_cog_stacked)

    print("building gallery in", OUT_DIR)

    # composites
    scene.plot_composite(
        ["red", "green", "blue"],
        figsize=FIGSIZE,
        title="True colour (red, green, blue)",
        save_path=OUT_DIR / "composite_true_colour.jpg",
    )
    scene.plot_composite(
        ["nir", "red", "green"],
        figsize=FIGSIZE,
        title="False colour (nir, red, green) - vegetation reads red",
        save_path=OUT_DIR / "composite_false_colour.jpg",
    )

    # index maps
    ndvi = scene.ndvi(red="red", nir="nir", name="NDVI")
    ndvi.plot_raster(
        cmap="RdYlGn",
        figsize=FIGSIZE,
        stretch=False,
        colorbar=True,
        save_path=OUT_DIR / "index_ndvi.jpg",
    )

    # dem
    dem = eeo.load_raster(sd.copernicus_dem, band_names=["elevation"])
    dem.plot_raster(
        cmap="Spectral_r",
        figsize=FIGSIZE,
        stretch=False,
        colorbar=True,
        title="Copernicus DEM over the same footprint (356-560 m)",
        save_path=OUT_DIR / "dem_terrain.jpg",
    )

    # histograms
    scene.plot_histogram(
        figsize=(7, 6),
        title="Band value distributions",
        save_path=OUT_DIR / "histogram_bands.png",
    )

    # clip to a vector AOI, then map + histogram together-
    clipped_ndvi = scene.clip_raster_with_vector(sd.boundary).ndvi(
        red="red", nir="nir", name="NDVI"
    )
    clipped_ndvi.plot_raster_with_histogram(
        cmap="RdYlGn",
        # wide enough that the UTM x-axis labels do not collide
        figsize=(13, 4.5),
        colorbar=True,
        title="NDVI clipped to a boundary polygon, with its distribution",
        save_path=OUT_DIR / "clip_ndvi_histogram.png",
    )

    print("shrinking for the web:")
    total = 0
    images = sorted(p for p in OUT_DIR.iterdir() if p.suffix in {".png", ".jpg"})
    for png in images:
        _shrink(png)
        total += png.stat().st_size
    print(f"gallery total: {total / 1e6:.2f} MB across {len(images)} images")


if __name__ == "__main__":
    main()
