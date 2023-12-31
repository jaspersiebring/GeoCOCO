# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import toml
import pathlib
from datetime import datetime

# Appending project root to PATH
sys.path.insert(0, os.path.abspath(".."))

# Getting package version from pyproject toml
meta_path = pathlib.Path(__file__).parents[1] / "pyproject.toml"
meta_info = toml.load(meta_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = meta_info["tool"]["poetry"]["name"]
copyright = f"{datetime.now().year}, Jasper Siebring"
author = "Jasper Siebring"
version = release = meta_info["tool"]["poetry"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
