"""
Sphinx documentation configuration for mmcli.
"""

import sys
import os

# Add the parent directory to path so we can import mmcli
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'mmcli'
copyright = '2024, Texas Instruments'
author = 'Texas Instruments'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# HTML output options
html_theme = 'alabaster'
# html_static_path = ['_static']  # Optional: add custom static files

templates_path = ['_templates']  # Optional: add custom templates

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
