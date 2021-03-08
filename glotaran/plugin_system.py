"""Functionality to register, initialize and retrieve glotaran plugins.

Since this module is imported at the root __init__.py file all other
glotaran imports should be used for typechecking only in the 'if TYPE_CHECKING' block.
This is to prevent issues with circular imports.
"""

from importlib import metadata


def load_plugins():
    """Initialize plugins registered under the entrypoint 'glotaran.plugins'.

    For an entry_point to be considered a glotaran plugin it just needs to start with
    'glotaran.plugins', which allows for an easy extendability.

    Currently used builtin entrypoints are:
    * glotaran.plugins.data_io
    * glotaran.plugins.model
    * glotaran.plugins.project_io
    """

    for entry_point_name, entry_points in metadata.entry_points().items():
        if entry_point_name.startswith("glotaran.plugins"):
            for entry_point in entry_points:
                entry_point.load()
