Citing Easy-EO
==============

If Easy-EO contributed to work you are publishing, please cite it. Citations
are how open-source research software earns the recognition that keeps it
maintained.

Citing the software
-------------------

Easy-EO is archived on Zenodo, and every release gets its own archived
snapshot. The DOI below is the **concept DOI**: it always resolves to the most
recent release, so it stays correct as new versions appear.

.. code-block:: bibtex

    @software{easy_eo,
      author    = {Botchwey, Thomas Burns},
      title     = {Easy-EO: chainable raster analysis for Earth Observation
                   in Python},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.21967655},
      url       = {https://github.com/Tommy-Burns/easy-eo}
    }

To cite the exact version you used instead, open the `Zenodo record
<https://doi.org/10.5281/zenodo.21967655>`_, select that release, and use its
own version DOI. Naming the version is worth doing in a methods section, where
reproducibility depends on which release produced the numbers.

``CITATION.cff`` in the repository carries the same metadata in machine-readable
form. GitHub reads it to render the **Cite this repository** button, and tools
such as ``cffconvert`` convert it to BibTeX, APA, and other formats.

Citing the sample data
----------------------

The bundled sample files reached through :func:`eeo.datasets.load_sample_dataset`
are archived separately, with their own DOI. Cite it if a figure or result in
your work was produced from them:

.. code-block:: bibtex

    @dataset{easy_eo_sample_data,
      author    = {Botchwey, Thomas Burns},
      title     = {Easy-EO sample data: Sentinel-2 and Copernicus DEM
                   subsets for testing and teaching},
      publisher = {Zenodo},
      year      = {2026},
      version   = {1},
      doi       = {10.5281/zenodo.21917533}
    }

This is a version DOI rather than a concept DOI, deliberately: a reproducible
result depends on the exact files, and a later deposit could change them.

Attribution for the underlying data
-----------------------------------

Citing the deposit does not replace the attribution the source data requires.
The sample files are derived from open Copernicus data, which must be credited
whenever the data or figures made from it are redistributed. Each file carries
the exact wording:

.. code-block:: python

    from eeo.datasets import load_sample_dataset

    sd = load_sample_dataset()
    print(sd.sentinel2_cog_stacked.attribution)

See :doc:`user_guide/sample_data` for the full terms.
