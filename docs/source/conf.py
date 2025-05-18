# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import toml
import pathlib
from datetime import datetime

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
meta_path = PROJECT_ROOT / "pyproject.toml"
meta_info = toml.load(meta_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = meta_info["project"]["name"]
release = version = release = meta_info["project"]["version"]
author = "Jasper Siebring"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser"
]
suppress_warnings = ["autoapi"]
autodoc_typehints = "description"
autoapi_own_page_level = "module"
autoapi_add_toctree_entry = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = []
autoapi_dirs = [str(PROJECT_ROOT / project)]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
