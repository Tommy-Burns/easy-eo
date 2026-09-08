Preprocessing Module
====================

.. automodule:: eeo.preprocessing
    :members:
    :undoc-members:
    :show-inheritance:

Default class and flag sets
---------------------------

The sets :func:`~eeo.mask_clouds` uses when it is not told otherwise. They are
named constants so that a user who disagrees can say so, and so a published
analysis can state exactly what it masked — see :doc:`../user_guide/masking_clouds`.

They are documented from the module that defines them, because their
descriptions live with the definitions rather than with the re-export.

.. autodata:: eeo.preprocessing.quality.SCL_CLOUDY
.. autodata:: eeo.preprocessing.quality.SCL_NODATA
.. autodata:: eeo.preprocessing.quality.SCL_DEFAULT_MASKED
.. autodata:: eeo.preprocessing.quality.QA_PIXEL_CLOUDY
.. autodata:: eeo.preprocessing.quality.QA_PIXEL_NODATA
.. autodata:: eeo.preprocessing.quality.QA_PIXEL_DEFAULT_MASKED
.. autodata:: eeo.preprocessing.quality.QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE
