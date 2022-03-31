"""The glotaran parameter registry module."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import load_parameters
from glotaran.io import save_parameters
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project.project_registry import ProjectRegistry


class ProjectParameterRegistry(ProjectRegistry):
    """A registry for parameters."""

    def __init__(self, directory: Path):
        """Initialize a parameter registry.

        Parameters
        ----------
        directory : Path
            The registry directory.
        """
        super().__init__(directory / "parameters", [".yml", ".yaml", ".csv"], load_parameters)

    def generate_parameters(
        self,
        model: Model,
        name: str | None,
        *,
        format_name: Literal["yml", "yaml", "csv"] = "csv",
        allow_overwrite: bool = False,
        ignore_existing: bool = False,
    ):
        """Generate parameters for a model.

        Parameters
        ----------
        model : Model
            The model.
        name : str | None
            The name of the parameters.
        format_name : Literal["yml", "yaml", "csv"]
            The parameter format.
        allow_overwrite: bool
            Whether to overwrite existing parameters.
        ignore_existing: bool
            Whether to ignore generation of a parameter file if it already exists.

        Raises
        ------
        FileExistsError
            Raised if parameters is already existing and `allow_overwrite=False`.
        """
        parameters = model.generate_parameters()
        parameter_file = self.directory / f"{name}.{format_name}"

        if parameter_file.exists() and ignore_existing:
            return

        if parameter_file.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Parameters {name!r} already exists and `allow_overwrite=False`."
            )
        if format_name in ["yml", "yaml"]:
            write_dict(parameters, file_name=parameter_file, offset=0)
        elif format_name == "csv":
            parameter_group = (
                ParameterGroup.from_dict(parameters)
                if isinstance(parameters, dict)
                else ParameterGroup.from_list(parameters)
            )
            save_parameters(parameter_group, parameter_file, allow_overwrite=allow_overwrite)
