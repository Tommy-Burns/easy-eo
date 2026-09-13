Core Module
===========

.. automodule:: eeo.core
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: EEOError, ValidationError, CRSMismatchError, AlignmentError, BackendError, MissingDependencyError

Utilities
---------

.. autofunction:: eeo.show_versions

Exceptions
----------

Easy-EO raises a small hierarchy of exceptions so you can catch precisely what
you need. Every library-specific error derives from :class:`~eeo.EEOError`, so a
single ``except`` clause handles any of them while letting unrelated errors
propagate:

.. code-block:: python

    import eeo

    try:
        ds = eeo.load_raster("scene.tif")
        result = ds.normalized_difference(other, auto_align=False)
    except eeo.AlignmentError as err:
        # rasters were not on the same grid
        print(err)
    except eeo.EEOError as err:
        # any other Easy-EO failure
        print(err)

For backward compatibility, each subclass also derives from the built-in
exception it historically replaced: :class:`~eeo.ValidationError`,
:class:`~eeo.CRSMismatchError`, and :class:`~eeo.AlignmentError` are
``ValueError``\ s, :class:`~eeo.BackendError` is a ``RuntimeError``, and
:class:`~eeo.MissingDependencyError` — raised when a feature's optional extra
is not installed — is an ``ImportError``. Two
failure modes intentionally keep their standard-library exceptions rather than
joining the hierarchy: a missing raster file raises ``FileNotFoundError``, and
an out-of-range band index raises ``IndexError``.

.. automodule:: eeo.core.exceptions
    :members:
    :show-inheritance:

Block-wise execution
--------------------

Pixel-wise operations do not have to hold a whole scene in memory. This module
iterates a raster in windows, calls a plain NumPy function on one window's
worth of every operand, and writes each result straight into the output, so
peak memory follows the block size rather than the scene.

It is the shared engine operations delegate to, and an extension point for
anyone writing their own pixel-wise operation. Pass ``save_path=`` to stream
the result to a file, which is the only route whose memory stays bounded when
the *output* is also larger than memory:

.. code-block:: python

    import numpy as np
    import eeo
    from eeo.core.blockwise import BlockSource, apply_blockwise

    scene = eeo.load_landsat("LC09_....tar", bands=["red", "nir08"])

    def ndvi(nir, red):
        nir, red = nir.astype("float32"), red.astype("float32")
        total = nir + red
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(total != 0, (nir - red) / total, np.float32(0))

    result = apply_blockwise(
        scene,
        ndvi,
        sources=[
            BlockSource.from_dataset(scene, band=2),
            BlockSource.from_dataset(scene, band=1),
        ],
        fractional=True,
        save_path="ndvi.tif",
    )

The function passed in must be element-wise — the value it computes for a
pixel may not depend on any other pixel — because blocks are computed
independently. An operation that needs a statistic over the whole raster
(a percentile stretch, a standardization) has to make a pass for the
statistic first and only then stream the result.

.. automodule:: eeo.core.blockwise
    :members:
    :show-inheritance:

Streaming reductions
--------------------

Where :mod:`eeo.core.blockwise` streams a *transformation* — one block in, one
block out — this module streams a *reduction*: the raster is read a window at a
time and collapsed to a few numbers, so a statistic can be taken over a scene
that does not fit in memory.

.. automodule:: eeo.core.streaming
    :members:
    :show-inheritance:

