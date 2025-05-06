# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "VisionSIM"
copyright = "VisionSIM developers, 2025"
author = "Sacha Jungerman"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.programoutput",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "click_extra.sphinx",
    "sphinxcontrib.video",
    "sphinx.ext.autosectionlabel",
]
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# -- Options for autodocs ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_type_aliases = {"npt.ArrayLike": "npt.ArrayLike"}
autodoc_mock_imports = ["bpy"]
autodoc_default_options = {
    "special-members": "__init__,__call__",
}
autodoc_member_order = "bysource"

# -- Options for autosectionlabel---------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = True
