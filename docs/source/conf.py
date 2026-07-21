# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath("../.."))


project = "Easy-EO"
copyright = "2025, Thomas Burns Botchwey"
author = "Thomas Burns Botchwey"
release = "0.1.0b1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

extensions.append("sphinx_copybutton")

# -- Napoleon (NumPy-style docstrings) ---------------------------------------
# Docstrings are NumPy style everywhere (see CODE_STYLE.md); disable the
# Google parser so only one style is recognised.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Intersphinx -------------------------------------------------------------
# Resolve type references in docstrings (numpy.ndarray, affine.Affine,
# rasterio.crs.CRS, ...) to the upstream documentation as clickable links.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# Optional: customize sidebar/navigation depth
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
}
html_static_path = ["_static"]
