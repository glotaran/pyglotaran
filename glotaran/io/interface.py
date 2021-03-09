from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from typing import Union

    import xarray as xr

    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup
    from glotaran.project import Result
    from glotaran.project import SavingOptions
    from glotaran.project import Scheme

    DataLoader = Callable[[str], Union[xr.Dataset, xr.DataArray]]
    DataWriter = Callable[[str, SavingOptions, xr.Dataset], None]


class DataIoInterface:
    """Baseclass for Data IO plugins."""

    def __init__(self, fmt: str) -> None:
        self.format = fmt

    def read_dataset(self, file_name: str) -> xr.Dataset | xr.DataArray:
        """'read_dataset' isn't implemented for this format."""
        raise NotImplementedError(
            f"""'read_dataset' isn't implemented for the format: {self.format!r}"""
        )

    def write_dataset(self, file_name: str, saving_options: SavingOptions, dataset: xr.Dataset):
        """'write_dataset' isn't implemented for this format."""
        raise NotImplementedError(
            f"""'write_dataset' isn't implemented for the format: {self.format!r}"""
        )

    def get_dataloader(self) -> DataLoader:
        """Retrieve implementation of the read_dataset functionality.

        This allows to get the proper help and autocomplete for the function,
        which is especially valuable if the function provides additional options.

        Returns
        -------
        DataLoader
            Function to load data a given format as :xarraydoc:`Dataset` or :xarraydoc:`DataArray`.
        """
        return self.read_dataset

    def get_datawriter(self) -> DataWriter:
        """Retrieve implementation of the write_dataset functionality.

        This allows to get the proper help and autocomplete for the function,
        which is especially valuable if the function provides additional options.

        Returns
        -------
        DataWriter
            Function to write :xarraydoc:`Dataset` to a given format.
        """
        return self.write_dataset


class ProjectIoInterface:
    """Baseclass for Project IO plugins."""

    def __init__(self, fmt: str) -> None:
        self.format = fmt

    def read_model(self, file_name: str) -> Model:
        raise NotImplementedError

    def write_model(self, file_name: str, model: Model):
        raise NotImplementedError

    def read_parameters(self, file_name: str) -> ParameterGroup:
        raise NotImplementedError

    def write_parameters(self, file_name: str, parameters: ParameterGroup):
        raise NotImplementedError

    def read_scheme(self, file_name: str) -> Scheme:
        raise NotImplementedError

    def write_scheme(self, file_name: str, scheme: Scheme):
        raise NotImplementedError

    def read_result(self, file_name: str) -> Result:
        raise NotImplementedError

    def write_result(self, file_name: str, saving_options: SavingOptions, result: Result):
        raise NotImplementedError
