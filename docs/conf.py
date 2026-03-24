"""Sphinx configuration for SDO CLV Pipeline documentation."""
import sdo_clv_pipeline

project = "SDO CLV Pipeline"
author = "Michael Palumbo"
version = sdo_clv_pipeline.__version__ if hasattr(sdo_clv_pipeline, "__version__") else "0.5.2"
release = version

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
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
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "assets/logo.png"
html_title = "SDO CLV Pipeline"

html_theme_options = {
    "light_css_variables": {},
    "dark_css_variables": {},
}

# Suppress warnings for missing type stubs in third-party libs
autodoc_mock_imports = [
    "sunpy",
    "astropy",
    "scipy",
    "skimage",
    "numba",
    "tqdm",
    "pandas",
    "matplotlib",
    "numpy",
]
