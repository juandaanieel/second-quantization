# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "second_quantization"
copyright = "2024, Quantum Tinkerers"
author = "Quantum Tinkerers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
try:
    from second_quantization._version import __version__

    version = __version__
    release = __version__
except ImportError:
    # If the package is not installed, use a default version
    version = "0.1.0"
    release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",  # Automatic documentation from docstrings
    "sphinx.ext.autosummary",  # Generate autodoc summaries
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx.ext.mathjax",  # Math support via MathJax
    "sphinx.ext.githubpages",  # Publish HTML docs in GitHub Pages
    # MyST extensions for Markdown and executable notebooks
    "myst_nb",  # MyST Notebook parser for executable content
    # "myst_parser",                  # MyST Markdown parser
    # Additional useful extensions
    "sphinx_copybutton",  # Add copy buttons to code blocks
    "sphinx_design",  # Design elements (cards, tabs, etc.)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames.
# source_suffix = {
#     ".rst": None,
#     ".md": "myst-nb",
#     ".ipynb": "myst-nb",
# }

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/quantum-tinkerers/second_quantization",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "path_to_docs": "docs",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
    },
    "navigation_with_keys": False,
    "show_navbar_depth": 2,
}

html_title = f"{project} v{version}"
html_static_path = ["_static"]

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
html_extra_path = []

# -- Options for MyST-NB -----------------------------------------------------

# MyST-NB configuration for executable content
nb_execution_mode = "auto"  # Execute notebooks during build
nb_execution_timeout = 60  # Timeout for notebook execution (seconds)
nb_execution_allow_errors = False  # Stop build on execution errors

# Cache executed notebooks to speed up builds
nb_execution_cache_path = "_build/.jupyter_cache"

# Kernel to use for executing notebooks
nb_kernel_rgx_aliases = {
    "python3": "python",
}

# Configure MyST parser
myst_enable_extensions = [
    "amsmath",  # LaTeX math
    "colon_fence",  # ::: code fences
    "deflist",  # Definition lists
    "dollarmath",  # $ and $$ math
    "html_image",  # HTML images
    "linkify",  # Auto-link URLs
    "replacements",  # Text replacements
    "smartquotes",  # Smart quotes
    "substitution",  # Variable substitutions
    "tasklist",  # Task lists
]

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Automatically extract typehints when generating documentation
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Generate autosummary pages
autosummary_generate = True

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for Napoleon (Google/NumPy style docstrings) -------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for Math --------------------------------------------------------

# Use MathJax for math rendering
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "processEnvironments": True,
    }
}

# -- Options for copy button -------------------------------------------------

# Configure copy button for code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_exclude = ".linenos, .gp, .go"

# -- Custom configuration ----------------------------------------------------

# Configure syntax highlighting
pygments_style = "sphinx"

# Configure line numbers for code blocks
html_codeblock_linenos_style = "table"

# Configure figure numbering
numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section %s",
}

# Configure cross-references
suppress_warnings = ["myst.header"]

# # -- Notebook execution configuration ----------------------------------------

# # # Configure Jupyter kernel for execution
# # jupyter_execute_notebooks = "auto"

# # # Configure plotting backend for matplotlib
# # nb_custom_formats = {
# #     ".md": ["jupytext", "--to", "myst"]
# # }


# # Default cell execution timeout
# nb_execution_timeout = 120

# # Show stderr output from notebook execution
# nb_output_stderr = "show"

# # Configure matplotlib for inline plots
# nb_render_plugin = "default"

# # Enable plot directive
# plot_include_source = True
# plot_html_show_source_link = False
# plot_formats = ["png", "svg"]
