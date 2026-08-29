IO Module
=========

Data access and exchange beyond the local filesystem: STAC search, loaders
for products downloaded to disk, and xarray interop. STAC and xarray each need
their optional extra (``pip install "easy-eo[stac]"`` /
``pip install "easy-eo[xarray]"``); without it, the call raises
:class:`~eeo.MissingDependencyError` with the install command. The product
loaders need no extra.

:func:`eeo.stac_search` finds scenes and :meth:`eeo.io.STACItem.load` reads
them. Searches go to Microsoft Planetary Computer
(:data:`eeo.io.PLANETARY_COMPUTER_STAC_URL`) unless a different ``catalog=`` is
given. A load returns only the area of interest by default — the assets are read
remotely over HTTP range requests, so a small AOI over a Sentinel-2 tile
transfers a fraction of the band instead of downloading the scene:

.. code-block:: python

    import eeo

    results = eeo.stac_search(
        "sentinel-2-l2a",
        bbox=(11.0, 46.5, 11.2, 46.7),
        datetime="2023-06-01/2023-08-31",
        cloud_cover=20,
        limit=1,
    )
    scene = results[0].load(["B04", "B08"])   # cropped to the search bbox
    ndvi = scene.ndvi(red="B04", nir="B08")

STAC search
-----------

.. autofunction:: eeo.stac_search

.. autoclass:: eeo.io.STACSearchResult
    :members:

.. autoclass:: eeo.io.STACItem
    :members:

.. autodata:: eeo.io.PLANETARY_COMPUTER_STAC_URL

Downloaded products
-------------------

:func:`eeo.load_sentinel2` reads a Copernicus ``.SAFE`` product and
:func:`eeo.load_landsat` a USGS Collection 2 Level-2 one — one function for
Landsat 4, 5, 7, 8, and 9, since the mission only selects which band table the
names resolve against. Either accepts the product unpacked or still in the
``.zip`` or ``.tar`` it was downloaded as, and both return the same dataset a
catalog load does:

.. code-block:: python

    import eeo

    scene = eeo.load_sentinel2(path="S2B_MSIL2A_….SAFE", bands=["red", "nir"])
    ndvi = scene.ndvi(red="red", nir="nir")

Values are the product's stored integers, not reflectance —
see :doc:`../user_guide/loading_downloaded_scenes`.

.. autofunction:: eeo.load_sentinel2

.. autofunction:: eeo.load_landsat

xarray interop
--------------

:func:`eeo.from_xarray` wraps a georeferenced :class:`xarray.DataArray` as a
dataset, and :meth:`eeo.core.core.EEORasterDataset.to_xarray` converts one back,
so data can cross into the xarray ecosystem and return:

.. code-block:: python

    import eeo
    import rioxarray

    ds = eeo.from_xarray(rioxarray.open_rasterio("scene.tif"))
    ndvi = ds.ndvi(red=1, nir=4)

    da = ndvi.to_xarray()          # back out, CRS and nodata intact
    da.rio.to_raster("ndvi.tif")

.. autofunction:: eeo.from_xarray
