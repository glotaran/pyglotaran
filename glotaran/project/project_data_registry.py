"""The glotaran data registry module."""
from __future__ import annotations

from pathlib import Path

from glotaran.io import load_dataset
from glotaran.plugin_system.data_io_registration import known_data_formats
from glotaran.project.project_registry import ProjectRegistry


class ProjectDataRegistry(ProjectRegistry):
    """A registry for data."""

    def __init__(self, directory: Path):
        """Initialize a data registry.

        Parameters
        ----------
        directory : Path
            The registry directory.
        """
        super().__init__(
            directory / "data", [f".{fmt}" for fmt in known_data_formats()], load_dataset
        )

    def import_data(self, path: str | Path, name: str | None = None):
        """Import a dataset.

        Parameters
        ----------
        path : str | Path
            The path to the dataset.
        name : str | None
            The name of the dataset.
        """
        if not isinstance(path, Path):
            path = Path(path)

        name = name or path.with_suffix("").name
        data_path = self.directory / f"{name}.nc"

        dataset = load_dataset(path)
        dataset.attrs.pop("loader")
        dataset.to_netcdf(data_path)