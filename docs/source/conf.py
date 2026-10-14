# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys

# Make the quantx_engine package importable from docs/source
sys.path.insert(0, os.path.abspath("../../python"))

# ── Project information ────────────────────────────────────────────────────────

project = "QuantX Engine"
copyright = "2026, QuantX Contributors"
author = "QuantX Contributors"
# The version is written once, in python/quantx_engine/__init__.py (see RELEASING.md). Read
# it rather than import the package, which would pull in every SDK module.
import re as _re

with open(os.path.join(os.path.dirname(__file__), "..", "..", "python", "quantx_engine",
                       "__init__.py"), encoding="utf-8") as _f:
    release = _re.search(r'^__version__ = "([^"]+)"', _f.read(), _re.M).group(1)
version = ".".join(release.split(".")[:2])

# ── General configuration ─────────────────────────────────────────────────────

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML output ────────────────────────────────────────────────────────────────

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"QuantX Engine {release}"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/ThisIsBarney/quantx",
    "source_branch": "master",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ThisIsBarney/quantx",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

# ── intersphinx ────────────────────────────────────────────────────────────────

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# ── Napoleon (Google/NumPy docstrings) ────────────────────────────────────────

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True   # Attributes sections as fields: dataclass members are also autodoc members, so attribute directives would document each one twice
napoleon_use_param = True
napoleon_use_rtype = True

# ── autodoc ───────────────────────────────────────────────────────────────────

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__init__",
    "inherited-members": False,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# autosummary auto-generates stub .rst files for each module
autosummary_generate = True
autosummary_imported_members = False

# ── todo ──────────────────────────────────────────────────────────────────────

todo_include_todos = True
