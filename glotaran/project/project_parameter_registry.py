"""The glotaran parameter registry module."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from yaml import dump

from glotaran.io import load_parameters
from glotaran.io import save_parameters
from glotaran.model import Model
from glotaran.model import ModelError
from glotaran.parameter import ParameterGroup
from glotaran.parameter.parameter import Keys
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
        fmt: Literal["yml", "yaml", "csv"] = "csv",
    ):
        """Generate parameters for a model.

        Parameters
        ----------
        model : Model
            The model.
        name : str | None
             The name of the parameters.
        fmt : Literal["yml", "yaml", "csv"]
            The parameter format.

        Raises
        ------
        ModelError
            Raised if parameter labels are incompatible.
        """
        parameters: dict | list = {}
        for parameter in model.get_parameter_labels():
            groups = parameter.split(".")
            label = groups.pop()
            if len(groups) == 0:
                if isinstance(parameters, dict):
                    if len(parameters) != 0:
                        raise ModelError(
                            "The root parameter group cannot contain both groups and parameters."
                        )
                    else:
                        parameters = []
                parameters.append(
                    [
                        label,
                        0.0,
                        {
                            Keys.EXPR: "None",
                            Keys.MAX: "None",
                            Keys.MIN: "None",
                            Keys.NON_NEG: "false",
                            Keys.VARY: "true",
                        },
                    ]
                )
            else:
                if isinstance(parameters, list):
                    raise ModelError(
                        "The root parameter group cannot contain both groups and parameters."
                    )
                this_group = groups.pop()
                group = parameters
                for name in groups:
                    if name not in group:
                        group[name] = {}
                    group = group[name]
                if this_group not in group:
                    group[this_group] = []
                group[this_group].append(
                    [
                        label,
                        0.0,
                        {
                            Keys.EXPR: None,
                            Keys.MAX: "inf",
                            Keys.MIN: "-inf",
                            Keys.NON_NEG: "false",
                            Keys.VARY: "true",
                        },
                    ]
                )

        parameter_file = self.directory / f"{name}.{fmt}"
        if fmt in ["yml", "yaml"]:
            parameter_yml = dump(parameters)
            with open(parameter_file, "w") as f:
                f.write(parameter_yml)
        elif fmt == "csv":
            parameter_group = (
                ParameterGroup.from_dict(parameters)
                if isinstance(parameters, dict)
                else ParameterGroup.from_list(parameters)
            )
            save_parameters(parameter_group, parameter_file)
