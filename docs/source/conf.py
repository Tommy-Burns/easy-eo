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
release = "0.5.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
]

extensions.append("sphinx_copybutton")
extensions.append("sphinx_design")

# -- matplotlib plot_directive -----------------------------------------------
# Render the spectral-index comparison figure from source at build time (no
# committed binary). Show only the image, not the source/PNG-vs-HiRes links.
plot_include_source = False
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 100)]

# -- Napoleon (NumPy-style docstrings) ---------------------------------------
# Docstrings are NumPy style everywhere (see CODE_STYLE.md); disable the
# Google parser so only one style is recognised.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_use_ivar = True

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
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# -- linkcheck ---------------------------------------------------------------
# Retry once more than the default, so a momentary timeout in the scheduled
# link check does not report a working link as broken.
linkcheck_retries = 2

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "Easy-EO"

# Top-level toctree entries in index.rst become the navbar tabs; each tab's
# own toctree fills the left sidebar. The header is always navy, so the logo
# is the light-outlined variant in both colour modes.
html_theme_options = {
    "logo": {
        "text": "Easy-EO",
        "image_light": "_static/logo/eeo-mark-on-dark.png",
        "image_dark": "_static/logo/eeo-mark-on-dark.png",
        "alt_text": "Easy-EO home",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tommy-burns/easy-eo",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/easy-eo/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "show_prev_next": True,
    "use_edit_page_button": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "github-light-colorblind",
    "pygments_dark_style": "github-dark-colorblind",
}
html_context = {
    "github_user": "tommy-burns",
    "github_repo": "easy-eo",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
}
# Single-page tabs have no section to navigate, so drop the empty sidebar.
html_sidebars = {"index": [], "getting_started": [], "tutorials": [], "citation": []}
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800"
    "&family=JetBrains+Mono:wght@400;600&display=swap",
    "css/eeo-theme.css",
]
html_favicon = "_static/logo/favicon.png"
