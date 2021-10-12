"""Baseclasses to create Data/Project IO plugins from.

The main purpose of those classes are to guarantee a consistent API via
typechecker like ``mypy`` and demonstarate with methods are accessed by
highlevel convenience functions for a given type of plugin.

To add additional options to a method, those options need to be
keyword only arguments.
See: https://www.python.org/dev/peps/pep-3102/

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from typing import Union

    import xarray as xr

    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup
    from glotaran.project import Result
    from glotaran.project import Scheme

    DataLoader = Callable[[str], Union[xr.Dataset, xr.DataArray]]
    DataSaver = Callable[[str, Union[xr.Dataset, xr.DataArray]], None]


class DataIoInterface:
    """Baseclass for Data IO plugins."""

    def __init__(self, format_name: str) -> None:
        """Initialize a Data IO plugin with the name of the format.

        Parameters
        ----------
        format_name : str
            Name of the supported format an instance uses.
        """
        self.format = format_name

    def load_dataset(self, file_name: str) -> xr.Dataset | xr.DataArray:
        """Read data from a file to :xarraydoc:`Dataset` or :xarraydoc:`DataArray` (**NOT IMPLEMENTED**).

        Parameters
        ----------
        file_name : str
            File containing the data.

        Returns
        -------
        xr.Dataset|xr.DataArray
            Data loaded from the file.


        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"""Cannot read data with format: {self.format!r}""")

    def save_dataset(
        self,
        dataset: xr.Dataset | xr.DataArray,
        file_name: str,
    ):
        """Save data from :xarraydoc:`Dataset` to a file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to be saved to file.
        file_name : str
            File to write the data to.


        .. # noqa: DAR101
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"""Cannot save data with format: {self.format!r}""")


class ProjectIoInterface:
    """Baseclass for Project IO plugins."""

    def __init__(self, format_name: str) -> None:
        """Initialize a Project IO plugin with the name of the format.

        Parameters
        ----------
        format_name : str
            Name of the supported format an instance uses.
        """
        self.format = format_name

    def load_model(self, file_name: str) -> Model:
        """Create a Model instance from the specs defined in a file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        file_name : str
            File containing the model specs.

        Returns
        -------
        Model
            Model instance created from the file.


        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"Cannot read models with format {self.format!r}")

    def save_model(self, model: Model, file_name: str):
        """Save a Model instance to a spec file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        model: Model
            Model instance to save to specs file.
        file_name : str
            File to write the model specs to.


        .. # noqa: DAR101
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"Cannot save models with format {self.format!r}")

    def load_parameters(self, file_name: str) -> ParameterGroup:
        """Create a ParameterGroup instance from the specs defined in a file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        file_name : str
            File containing the parameter specs.

        Returns
        -------
        ParameterGroup
            ParameterGroup instance created from the file.


        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"Cannot read parameters with format {self.format!r}")

    def save_parameters(self, parameters: ParameterGroup, file_name: str):
        """Save a ParameterGroup instance to a spec file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        parameters : ParameterGroup
            ParameterGroup instance to save to specs file.
        file_name : str
            File to write the parameter specs to.


        .. # noqa: DAR101
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"Cannot save parameters with format {self.format!r}")

    def load_scheme(self, file_name: str) -> Scheme:
        """Create a Scheme instance from the specs defined in a file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        file_name : str
            File containing the parameter specs.

        Returns
        -------
        Scheme
            Scheme instance created from the file.

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"Cannot read scheme with format {self.format!r}")

    def save_scheme(self, scheme: Scheme, file_name: str):
        """Save a Scheme instance to a spec file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        scheme : Scheme
            Scheme instance to save to specs file.
        file_name : str
            File to write the scheme specs to.


        .. # noqa: DAR101
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"Cannot save scheme with format {self.format!r}")

    def load_result(self, result_path: str) -> Result:
        """Create a Result instance from the specs defined in a file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        result_path : str
            Path containing the result data.

        Returns
        -------
        Result
            Result instance created from the file.


        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"Cannot read result with format {self.format!r}")

    def save_result(self, result: Result, result_path: str) -> list[str] | None:
        """Save a Result instance to a spec file (**NOT IMPLEMENTED**).

        Parameters
        ----------
        result : Result
            Result instance to save to specs file.
        result_path : str
            Path to write the result data to.


        .. # noqa: DAR101
        .. # noqa: DAR401
        """
        raise NotImplementedError(f"Cannot save result with format {self.format!r}")
