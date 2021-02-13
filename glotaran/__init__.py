"""Glotaran package __init__.py"""

from glotaran.io import load_model
from glotaran.io import load_parameters

__version__ = "0.3.0"


import pkg_resources

for entry_point in pkg_resources.iter_entry_points("glotaran.plugins"):
    entry_point.load()
