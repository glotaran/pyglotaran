"""Glotaran package __init__.py"""

from importlib import metadata

for entry_point in metadata.entry_points()["glotaran.plugins"]:
    entry_point.load()

__version__ = "0.3.0"
# TODO: add git SHA1 information
