# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "alphadia"
copyright = "2024, Mann Labs"
author = "Mann Labs"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    # 'sphinx.ext.autodoc',
    "autodocsumm",
    "nbsphinx",
    "myst_parser",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_custom_sections = ["Schema"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "light_logo": "logo/alphadia.png",
    "dark_logo": "logo/alphadia.png",
    "sidebar_hide_name": True,
}

autodoc_default_options = {
    "autosummary": True,
    "special-members": "__init__",  # Include __init__ methods.
}
