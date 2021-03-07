"""Glotaran package __init__.py"""

from importlib import metadata

for entry_point_name, entry_points in metadata.entry_points().items():
    if entry_point_name.startswith("glotaran.plugins"):
        for entry_point in entry_points:
            entry_point.load()

__version__ = "0.3.2"
# TODO: add git SHA1 information
