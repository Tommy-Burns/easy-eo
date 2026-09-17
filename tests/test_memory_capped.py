"""WP-17's acceptance test: a full-scene NDVI inside a memory-capped process.

The claim is that a raster far larger than the memory a process is allowed can
still be processed end to end. A cap makes that falsifiable: the same scene,
the same cap, and the whole-array form of the same arithmetic must fail where
the streamed form succeeds. Without that second half the test would only show
that a small scene fits.

The scene here is synthetic, because the claim is about size rather than about
any mission's data; ``test_real_scenes.py`` runs the same acceptance check on
both real products.
"""

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.windows import Window

from memory_harness import requires_capping, run_capped

#: 8000 x 8000, two uint16 bands: 268 MB on disk, and an eager NDVI of it needs
#: three 244 MB float32 arrays, which is what the cap below denies.
SIDE = 8000

#: Address-space cap for the acceptance runs. Measured on this scene: streaming
#: to disk peaks at about 490 MiB and still runs under a 900 MiB cap, while the
#: whole-array form fails below 1500 MiB. 1400 leaves margin on both sides.
CAP_MIB = 1400

#: What the streamed run must stay under. It peaks at ~490 MiB, of which 256 is
#: GDAL's pinned block cache, so this is generous and still far below the file.
PEAK_BUDGET_MIB = 800


@pytest.fixture(scope="module")
def big_scene(tmp_path_factory):
    """A two-band scene written a strip at a time, so building it is bounded too."""
    requires_capping()
    path = tmp_path_factory.mktemp("capped") / "scene.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=SIDE,
        width=SIDE,
        count=2,
        dtype="uint16",
        crs="EPSG:32633",
        transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
        nodata=0,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        for row in range(0, SIDE, 512):
            height = min(512, SIDE - row)
            red = (np.arange(height * SIDE, dtype="uint16").reshape(height, SIDE) + row) % 9000 + 1
            window = Window(0, row, SIDE, height)
            dst.write(red.astype("uint16"), 1, window=window)
            dst.write((red // 2 + 100).astype("uint16"), 2, window=window)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "nir")
    return path


@pytest.fixture(params=[None, 1024], ids=["rasterio", "lazy"])
def chunks(request):
    """Run the acceptance test on each backend in turn.

    The claim is about streaming, which both backends reach the same way, so
    neither is allowed to be the one it happens to hold for. The lazy half
    needs the extra.
    """
    if request.param is not None:
        pytest.importorskip("dask.array")
        pytest.importorskip("rioxarray")
    return request.param


def _reference_ndvi(path, window):
    """NDVI of one window, computed straight from the file with NumPy."""
    with rio.open(path) as src:
        red = src.read(1, window=window).astype("float32")
        nir = src.read(2, window=window).astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    ndvi[(red == 0) | (nir == 0)] = np.nan
    return ndvi


def test_the_cap_is_real(tmp_path):
    """Guards every test below: a cap that does not bite proves nothing."""
    run = run_capped(
        f"""
        import numpy as np
        np.zeros({CAP_MIB * 2} * 1024 * 1024, dtype="uint8")
        report(allocated=True)
        """,
        cap_mib=CAP_MIB,
    )

    assert not run.ok
    assert run.out_of_memory, run.stderr


def test_full_scene_ndvi_completes_under_the_cap(big_scene, tmp_path, chunks):
    """The acceptance test: the whole scene, streamed to disk, inside the cap."""
    out = tmp_path / "ndvi.tif"
    run = run_capped(
        f"""
        import eeo
        from eeo.core.blockwise import BlockSource, apply_blockwise
        import numpy as np

        def ndvi(nir, red):
            nir = nir.astype("float32")
            red = red.astype("float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                return (nir - red) / (nir + red)

        ds = eeo.load_raster({str(big_scene)!r}, chunks={chunks!r})
        result = apply_blockwise(
            ds,
            ndvi,
            sources=[BlockSource.from_dataset(ds, band=2), BlockSource.from_dataset(ds, band=1)],
            fractional=True,
            save_path={str(out)!r},
        )
        report(shape=list(result.get_shape()), dtype=str(result.get_metadata()["dtype"]))
        """,
        cap_mib=CAP_MIB,
    )

    assert run.ok, run.failure
    assert run.result["shape"] == [SIDE, SIDE]
    assert run.result["dtype"] == "float32"
    assert run.peak_rss_mib < PEAK_BUDGET_MIB, f"peaked at {run.peak_rss_mib:.0f} MiB"

    # The answer is right, not merely produced: check windows at the corners
    # and across the middle against NumPy on the same pixels.
    with rio.open(out) as saved:
        for window in [
            Window(0, 0, 64, 64),
            Window(SIDE - 64, SIDE - 64, 64, 64),
            Window(3000, 4000, 128, 96),
        ]:
            np.testing.assert_allclose(
                saved.read(1, window=window), _reference_ndvi(big_scene, window), equal_nan=True
            )


def test_the_whole_array_form_fails_under_the_same_cap(big_scene):
    """The other half of the claim: the cap really does deny the eager route."""
    run = run_capped(
        f"""
        import numpy as np, rasterio
        with rasterio.open({str(big_scene)!r}) as src:
            red = src.read(1).astype("float32")
            nir = src.read(2).astype("float32")
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
        report(mean=float(np.nanmean(ndvi)))
        """,
        cap_mib=CAP_MIB,
    )

    assert not run.ok, "the eager form fitted; the cap no longer discriminates"
    assert run.out_of_memory, run.stderr


def test_the_dataset_api_reaches_the_same_answer(big_scene, tmp_path, chunks):
    """``ds.ndvi(...)`` needs room for its result, and must still agree."""
    out = tmp_path / "op.tif"
    run = run_capped(
        f"""
        import eeo
        ds = eeo.load_raster({str(big_scene)!r}, chunks={chunks!r})
        ds.ndvi("red", nir="nir").save_raster({str(out)!r})
        report(done=True)
        """,
        # The chainable op returns the result as an in-memory raster, so this
        # run is bounded by the output rather than by the block: ~1.2 GiB
        # against the streamed 0.5 GiB. That is the documented difference
        # between the two routes, not a regression.
        cap_mib=3000,
    )

    assert run.ok, run.failure
    window = Window(1000, 2000, 128, 128)
    with rio.open(out) as saved:
        np.testing.assert_allclose(
            saved.read(1, window=window), _reference_ndvi(big_scene, window), equal_nan=True
        )
