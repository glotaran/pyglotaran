"""The glotaran data registry module."""
from __future__ import annotations

from pathlib import Path

import xarray as xr

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
        dataset: str | Path | xr.Dataset | xr.DataArray,
        dataset_name: str | None = None,
        allow_overwrite: bool = False,
        ignore_existing: bool = False,
    ):
        """Import a dataset.

        Parameters
        ----------
        dataset : str | Path | xr.Dataset | xr.DataArray
            Dataset instance or path to a dataset.
        dataset_name : str | None
            The name of the dataset (needs to be provided when dataset is an xarray instance).
            Defaults to None.
        allow_overwrite: bool
            Whether to overwrite an existing dataset.
        ignore_existing: bool
            Whether to ignore import if the dataset already exists.

        Raises
        ------
        ValueError
            When importing from xarray object and not providing a name.
        """
        if isinstance(dataset, (xr.DataArray, xr.Dataset)) and dataset_name is None:
            raise ValueError(
                "When importing data from a 'xarray.Dataset' or 'xarray.DataArray' "
                "it is required to also pass a ``dataset_name``."
            )
        if isinstance(dataset, xr.DataArray):
            dataset = dataset.to_dataset(name="data")

        if isinstance(dataset, (str, Path)):
            dataset = Path(dataset)

            if dataset.is_absolute() is False:
                dataset = (self.directory.parent / dataset).resolve()

            dataset_name = dataset_name or dataset.stem
            dataset = load_dataset(dataset)

        data_path = self.directory / f"{dataset_name}.nc"
        if data_path.exists() and ignore_existing and allow_overwrite is False:
            return
        save_dataset(dataset, data_path, allow_overwrite=allow_overwrite)
