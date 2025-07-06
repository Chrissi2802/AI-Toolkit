# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import ai_toolkit  # noqa: E402


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AI-Toolkit"
copyright = "2025, Chrissi"
author = ai_toolkit.get_author()
release = ai_toolkit.get_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Automatically generate documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google-style and NumPy-style docstrings
    "sphinx.ext.viewcode",  # Add links to source code in the documentation
    "sphinx.ext.coverage",  # Measure code coverage of the documentation
    "sphinx.ext.mathjax",  # Support for rendering LaTeX math in the documentation
    "sphinx.ext.inheritance_diagram",  # Generate inheritance diagrams for classes
    "sphinx.ext.graphviz",  # Support for rendering Graphviz diagrams
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
