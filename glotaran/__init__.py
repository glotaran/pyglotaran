"""Glotaran package __init__.py"""

import pkg_resources

for entry_point in pkg_resources.iter_entry_points("glotaran.plugins"):
    entry_point.load()

__version__ = "0.3.0"
# TODO: add git SHA1 information
