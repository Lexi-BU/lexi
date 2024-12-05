import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LEXI"
copyright = "2024, Ramiz Qudsi, Brian Walsh, Cadin Connor"
author = "Ramiz Qudsi, Brian Walsh, Cadin Connor"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]  # Ensure _static is in the static path
html_css_files = [
    "css/custom.css",  # Add custom.css to the list
]
html_logo = "_static/lexi_logo.png"


html_context = {
    "display_github": True,  # Enable the "View page source" link
    "github_user": "Lexi-BU",  # GitHub username or organization name
    "github_repo": "lexi",  # Repository name
    "github_version": "stable",  # Branch name
    "conf_py_path": "/lexi/",  # Path to the relevant file directory
    "source_suffix": ".py",  # Set to Python file extension
}

# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": True,
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Options for autodock mock imports ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports

autodoc_mock_imports = [
    "numpy",
    "pandas",
    "matplotlib",
    "astropy",
    "scipy",
    "cdflib",
    "jupyter",
    "lexi.__init__",
    "lexi.__version__",
]

html_theme_options = {
    "description": "Documentation for the LEXI Package",
    "sidebar_collapse": False,
    "page_width": "80%",
    "fixed_sidebar": True,
}
