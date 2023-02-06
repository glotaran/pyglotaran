"""The glotaran data registry module."""
from __future__ import annotations

from pathlib import Path

from glotaran.io import load_dataset
from glotaran.io import save_dataset
from glotaran.plugin_system.data_io_registration import supported_file_extensions_data_io
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
            directory / "data",
            supported_file_extensions_data_io("load_dataset"),
            load_dataset,
            item_name="Dataset",
        )

    def import_data(
        self,
        path: str | Path,
        name: str | None = None,
        allow_overwrite: bool = False,
        ignore_existing: bool = False,
    ):
        """Import a dataset.

        Parameters
        ----------
        path : str | Path
            The path to the dataset.
        name : str | None
            The name of the dataset.
        allow_overwrite: bool
            Whether to overwrite an existing dataset.
        ignore_existing: bool
            Whether to ignore import if the dataset already exists.
        """
        path = Path(path)

        if path.is_absolute() is False:
            path = (self.directory.parent / path).resolve()

        name = name or path.stem
        data_path = self.directory / f"{name}.nc"

        if data_path.exists() and ignore_existing:
            return

        dataset = load_dataset(path)
        save_dataset(dataset, data_path, allow_overwrite=allow_overwrite)
