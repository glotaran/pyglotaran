from __future__ import annotations

from dataclasses import dataclass
from os import getcwd
from os import mkdir
from pathlib import Path
from typing import Any
from typing import Literal

import xarray as xr
from yaml import dump
from yaml import load

from glotaran import __version__ as gta_version
from glotaran.analysis.optimize import optimize
from glotaran.io import load_dataset
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.io import save_scheme
from glotaran.model import Model
from glotaran.model import ModelError
from glotaran.parameter import ParameterGroup
from glotaran.parameter.parameter import Keys
from glotaran.project.generators.generator import available_generators
from glotaran.project.generators.generator import generate_model_yml
from glotaran.project.scheme import Scheme

TEMPLATE = """version: {gta_version}

name: {name}
"""

PROJECT_FILE_NAME = "project.gta"


@dataclass
class Project:
    """A project represents a projectfolder on disk which contains a project file.

    A projectfile is a file in `yml` format with name `project.gta`

    """

    file: Path
    name: str
    version: str

    folder: Path

    def __post_init__(self):
        if isinstance(self.file, str):
            self.file = Path(self.file)
        if self.folder is None:
            self.folder = self.file.parent
        if isinstance(self.folder, str):
            self.folder = Path(self.folder)
        pass

    @classmethod
    def create(cls, name: str | None = None, folder: str | Path | None = None) -> Project:
        """Creates a new project.

        Parameters
        ----------
            name : str | None
                The name of the project. If ``None``, the name of the project folder will be used.
            folder : str | Path | None
                The folder where the project will be created. If ``None``, the current work
                directory will be used.

        Returns
        -------
        Project :
            The created project.

        """
        if folder is None:
            folder = getcwd()
        project_folder = Path(folder)
        name = name if name else project_folder.name
        project_file = project_folder / PROJECT_FILE_NAME
        with open(project_file, "w") as f:
            f.write(TEMPLATE.format(gta_version=gta_version, name=name))

        return cls.open(project_file)

    @classmethod
    def open(cls, project_folder_or_file: str | Path):
        folder = Path(project_folder_or_file)
        if folder.is_dir():
            file = folder / PROJECT_FILE_NAME
        else:
            folder, file = folder.parent, folder

        with open(file) as f:
            project_dict = load(f)
        project_dict["file"] = file
        project_dict["folder"] = folder
        return cls(**project_dict)

    @property
    def data_dir(self) -> Path:
        return self.folder / "data/"

    def create_data_dir_if_not_exist(self):
        if not self.data_dir.exists():
            mkdir(self.data_dir)

    @property
    def has_data(self) -> bool:
        return len(self.data) != 0

    @property
    def data(self):
        if not self.data_dir.exists():
            return {}
        return {
            data_file.with_suffix("").name: data_file
            for data_file in self.data_dir.iterdir()
            if data_file.suffix == ".nc"
        }

    def load_data(self, name: str) -> xr.Dataset | xr.DataArray:
        try:
            data_path = next(p for p in self.data_dir.iterdir() if name in p.name)
        except StopIteration:
            raise ValueError(f"Model file for model '{name}' does not exist.")
        return load_dataset(data_path)

    def import_data(self, path: str | Path, name: str | None = None):

        if not isinstance(path, Path):
            path = Path(path)

        name = name or path.with_suffix("").name
        data_path = self.data_dir / f"{name}.nc"

        self.create_data_dir_if_not_exist()
        dataset = load_dataset(path)
        dataset.to_netcdf(data_path)

    @property
    def model_dir(self) -> Path:
        return self.folder / "models/"

    def create_model_dir_if_not_exist(self):
        if not self.model_dir.exists():
            mkdir(self.model_dir)

    @property
    def has_models(self) -> bool:
        return len(self.models) != 0

    @property
    def models(self):
        if not self.model_dir.exists():
            return {}
        return {
            model_file.with_suffix("").name: model_file
            for model_file in self.model_dir.iterdir()
            if model_file.suffix in [".yml", ".yaml"]
        }

    def load_model(self, name: str) -> Model:
        model_path = self.model_dir / f"{name}.yml"
        if not model_path.exists():
            raise ValueError(f"Model file for model '{name}' does not exist.")
        return load_model(model_path)

    def generate_model(
        self,
        name: str,
        generator: str,
        generator_arguments: dict[str, Any],
    ):
        if generator not in available_generators:
            raise ValueError(f"Unknown generator '{generator}'.")
        self.create_model_dir_if_not_exist()
        model = generate_model_yml(generator, **generator_arguments)
        with open(self.model_dir / f"{name}.yml", "w") as f:
            f.write(model)

    @property
    def scheme_dir(self) -> Path:
        return self.folder / "schemes/"

    def create_scheme_dir_if_not_exist(self):
        if not self.scheme_dir.exists():
            mkdir(self.scheme_dir)

    @property
    def has_schemes(self) -> bool:
        return len(self.schemes) != 0

    @property
    def schemes(self):
        if not self.scheme_dir.exists():
            return {}
        return {
            scheme_file.with_suffix("").name: scheme_file
            for scheme_file in self.scheme_dir.iterdir()
            if scheme_file.suffix in [".yml", ".yaml"]
        }

    def load_scheme(self, name: str) -> Scheme:
        scheme_path = self.scheme_dir / f"{name}.yml"
        if not scheme_path.exists():
            raise ValueError(f"Scheme file for scheme '{name}' does not exist.")
        return load_scheme(scheme_path)

    def create_scheme(
        self,
        model: str,
        parameter: str,
        name: str | None = None,
        nfev: int = None,
        nnls: bool = False,
    ):

        self.create_scheme_dir_if_not_exist()
        if name is None:
            n = 1
            name = "scheme-1"
            scheme_path = self.scheme_dir / f"{name}.yml"
            while scheme_path.exists():
                n += 1
                scheme_path = self.scheme_dir / f"scheme-{n}.yml"
        else:
            scheme_path = self.scheme_dir / f"{name}.yml"

        models = self.models
        if model not in models:
            raise ValueError(f"Unknown model '{model}'")
        model = str(models[model])

        parameters = self.parameters
        if parameter not in parameters:
            raise ValueError(f"Unknown parameter '{parameter}'")
        parameter = str(parameters[parameter])

        data = self.data
        datasets = {}
        for dataset in load_model(model).dataset:  # type: ignore
            if dataset not in data:
                raise ValueError(f"Data missing for dataset '{dataset}'")
            datasets[dataset] = str(data[dataset])

        #  scheme = Scheme(
        #      model,
        #      parameter,
        #      datasets,
        #      non_negative_least_squares=nnls,
        #      maximum_number_function_evaluations=nfev,
        #  )
        #  save_scheme(scheme, scheme_path)

    @property
    def parameters_dir(self) -> Path:
        return self.folder / "parameters/"

    def create_parameters_dir_if_not_exist(self):
        if not self.parameters_dir.exists():
            mkdir(self.parameters_dir)

    @property
    def has_parameters(self) -> bool:
        return len(self.parameters) != 0

    @property
    def parameters(self):
        if not self.parameters_dir.exists():
            return {}
        return {
            parameters_file.with_suffix("").name: parameters_file
            for parameters_file in self.parameters_dir.iterdir()
            if parameters_file.suffix in [".yml", ".yaml", ".csv"]
        }

    def load_parameters(self, name: str) -> ParameterGroup:
        try:
            parameters_path = next(p for p in self.parameters_dir.iterdir() if name in p.name)
        except StopIteration:
            raise ValueError(f"Parameters file for parameters '{name}' does not exist.")
        return load_parameters(parameters_path)

    def generate_parameters(
        self,
        model_name: str,
        name: str | None = None,
        fmt: Literal["yml", "yaml", "csv"] = "csv",
    ):
        self.create_parameters_dir_if_not_exist()
        model = self.load_model(model_name)
        parameters: dict | list = {}
        for parameter in model.get_parameters():
            groups = parameter.split(".")
            label = groups.pop()
            if len(groups) == 0:
                if isinstance(parameters, dict) and len(parameters) != 0:
                    raise ModelError(
                        "The root parameter group cannot contain both groups and parameters."
                    )
                elif isinstance(parameters, dict):
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

        name = name if name is not None else model_name + "_parameters"
        parameter_file = self.parameters_dir / f"{name}.{fmt}"
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
            parameter_group.to_csv(parameter_file)

    def run(self, scheme_name: str):
        schemes = self.schemes
        if scheme_name not in schemes:
            raise ValueError(f"Unknown scheme {scheme_name}.")
        scheme = self.load_scheme(scheme_name)

        optimize(scheme)
