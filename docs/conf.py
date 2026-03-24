"""Sphinx configuration for SDO CLV Pipeline documentation."""
import os

project = "SDO CLV Pipeline"
author = "Michael Palumbo"
version = "0.5.2"
release = version

# Set by CI workflow to match the gh-pages directory name (dev, v0.5.2, etc.)
# Falls back to "dev" for local builds.
docs_version = os.environ.get("DOCS_VERSION", "dev")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Theme
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "assets/logo.png"
html_title = "SDO CLV Pipeline"

html_theme_options = {
    "repository_url": "https://github.com/palumbom/sdo-clv-pipeline",
    "use_repository_button": True,
    "show_toc_level": 2,
    "use_fullscreen_button": False,
    "home_page_in_toc": True,
    "switcher": {
        "json_url": "https://palumbom.github.io/sdo-clv-pipeline/switcher.json",
        "version_match": docs_version,
    },
    "check_switcher": False,
    "navbar_start": [],
    "primary_sidebar_end": ["version-switcher"],
}

exclude_patterns = ["_build", "superpowers"]
