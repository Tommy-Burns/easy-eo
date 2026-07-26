IO Module
=========

Data access beyond the local filesystem. STAC search needs the optional
``stac`` extra (``pip install "easy-eo[stac]"``); without it, calling
:func:`eeo.stac_search` raises :class:`~eeo.MissingDependencyError` with the
install command.

:func:`eeo.stac_search` finds scenes and :meth:`eeo.io.STACItem.load` reads
them. A load returns only the area of interest by default — the assets are read
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
