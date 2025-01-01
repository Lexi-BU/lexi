import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = "LEXI"
copyright = "LEXI Team @ Boston University, 2024"
author = "Ramiz Qudsi, Brian Walsh, Cadin Connor"

html_baseurl = "https://lexi-bu.github.io/"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autosummary_imported_members = True
autodoc_typehints = "description"

templates_path = ["_templates"]
html_static_path = ["_static"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_css_files = [
    "css/custom.css",
]
html_logo = "src/_static/lexi_logo.png"

html_context = {
    "display_github": True,
    "github_user": "Lexi-BU",
    "github_repo": "lexi",
    "github_version": "stable",
    "conf_py_path": "/docs/",
}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": True,
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_mock_imports = [
    "numpy",
    "pandas",
    "matplotlib",
    "astropy",
    "scipy",
    "cdflib",
    "jupyter",
]

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
}
