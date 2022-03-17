"""The glotaran project module."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Literal

import xarray as xr

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project.project_data_registry import ProjectDataRegistry
from glotaran.project.project_model_registry import ProjectModelRegistry
from glotaran.project.project_parameter_registry import ProjectParameterRegistry
from glotaran.project.project_result_registry import ProjectResultRegistry
from glotaran.project.result import Result
from glotaran.project.scheme import Scheme
from glotaran.utils.io import make_path_absolute_if_relative
from glotaran.utils.ipython import MarkdownStr

TEMPLATE = """version: {gta_version}
"""

PROJECT_FILE_NAME = "project.gta"


@dataclass
class Project:
    """A project represents a projectfolder on disk which contains a project file.

    A project file is a file in `yml` format with name `project.gta`
    """

    version: str = field(init=False)

    file: Path
    folder: Path | None = field(default=None)

    def __post_init__(self):
        """Overwrite of post init."""
        self.file = Path(self.file)
        if self.folder is None:
            self.folder = self.file.parent
        self.folder = Path(self.folder)

        self._data_registry = ProjectDataRegistry(self.folder)
        self._model_registry = ProjectModelRegistry(self.folder)
        self._parameter_registry = ProjectParameterRegistry(self.folder)
        self._result_registry = ProjectResultRegistry(self.folder)

    @staticmethod
    def create(folder: str | Path, overwrite: bool = False):
        """Create a new project folder and file.

        Parameters
        ----------
        folder : str | Path | None
            The folder where the project will be created. If ``None``, the current work
            directory will be used.
        overwrite: bool
            Whether to overwrite an existing project file.

        Raises
        ------
        FileExistsError
            Raised if the project file already exists and `overwrite=False`.
        """
        from glotaran import __version__ as gta_version

        project_folder = make_path_absolute_if_relative(Path(folder))
        project_folder.mkdir(parents=True, exist_ok=True)
        project_file = project_folder / PROJECT_FILE_NAME
        if project_file.exists() and not overwrite:
            raise FileExistsError(
                f"Project file '{project_file}' already exist. Set `overwrite=True` to overwrite."
            )
        project_file.write_text(TEMPLATE.format(gta_version=gta_version))

    @classmethod
    def open(cls, project_folder_or_file: str | Path, create_if_not_exist: bool = True):
        """Open a new project.

        Parameters
        ----------
        project_folder_or_file : str | Path
            The path to a project folder or file.
        create_if_not_exist : bool
            Create the project if not existent.

        Returns
        -------
        Project
            The created project.

        Raises
        ------
        FileNotFoundError
            Raised when the project file does not not exist and `create_if_not_exist` is `False`.
        """
        folder = make_path_absolute_if_relative(Path(project_folder_or_file))
        if folder.name == PROJECT_FILE_NAME:
            folder, file = folder.parent, folder
        else:
            file = folder / PROJECT_FILE_NAME

        if not file.exists():
            if not create_if_not_exist:
                raise FileNotFoundError(f"Project file {file} does not exists.")
            Project.create(folder)

        project_dict = load_dict(file, True)
        project_dict["file"] = file
        project_dict["folder"] = folder
        version = project_dict.pop("version")
        project = cls(**project_dict)
        project.version = version
        return project

    @property
    def has_data(self) -> bool:
        """Check if the project has datasets.

        Returns
        -------
        bool
            Whether the project has datasets.
        """
        return not self._data_registry.empty

    @property
    def data(self) -> dict[str, str]:
        """Get all project datasets.

        Returns
        -------
        dict[str, str]
            The models of the datasets.
        """
        return self._data_registry.items

    def load_data(self, name: str) -> xr.Dataset | xr.DataArray:
        """Load a dataset.

        Parameters
        ----------
        name : str
            The name of the dataset.

        Returns
        -------
        Result
            The loaded dataset.

        Raises
        ------
        ValueError
            Raised if the dataset does not exist.
        """
        try:
            return self._data_registry.load_item(name)
        except ValueError:
            raise ValueError(f"Dataset '{name}' does not exist.")

    def import_data(self, path: str | Path, name: str | None = None):
        """Import a dataset.

        Parameters
        ----------
        path : str | Path
            The path to the dataset.
        name : str | None
            The name of the dataset.
        """
        self._data_registry.import_data(path, name=name)

    @property
    def has_models(self) -> bool:
        """Check if the project has models.

        Returns
        -------
        bool
            Whether the project has models.
        """
        return not self._model_registry.empty

    @property
    def models(self) -> dict[str, str]:
        """Get all project models.

        Returns
        -------
        dict[str, str]
            The models of the project.
        """
        return self._model_registry.items

    def load_model(self, name: str) -> Model:
        """Load a model.

        Parameters
        ----------
        name : str
            The name of the model.

        Returns
        -------
        Model
            The loaded model.

        Raises
        ------
        ValueError
            Raised if the model does not exist.
        """
        try:
            return self._model_registry.load_item(name)
        except ValueError:
            raise ValueError(f"Model '{name}' does not exist.")

    def generate_model(
        self,
        generator_name: str,
        generator: str,
        generator_arguments: dict[str, Any],
    ):
        """Generate a model.

        Parameters
        ----------
        generator_name : str
            The name of the model.
        generator : str
            The generator for the model.
        generator_arguments : dict[str, Any]
            Arguments for the generator.
        """
        self._model_registry.generate_model(generator_name, generator, generator_arguments)

    def get_models_directory(self) -> Path:
        """Get the path to the model directory of the project.

        Returns
        -------
        Path
            The path to the project's model directory.
        """
        return self._model_registry.directory

    @property
    def has_parameters(self) -> bool:
        """Check if the project has parameters.

        Returns
        -------
        bool
            Whether the project has parameters.
        """
        return not self._parameter_registry.empty

    @property
    def parameters(self) -> dict[str, str]:
        """Get all project parameters.

        Returns
        -------
        dict[str, str]
            The parameters of the project.
        """
        return self._parameter_registry.items

    def load_parameters(self, name: str) -> ParameterGroup:
        """Load parameters.

        Parameters
        ----------
        name : str
            The name of the parameters.

        Returns
        -------
        ParameterGroup
            The loaded parameters.

        Raises
        ------
        ValueError
            Raised if parameters do not exist.
        """
        try:
            return self._parameter_registry.load_item(name)
        except ValueError:
            raise ValueError(f"Parameters '{name}' does not exist.")

    def generate_parameters(
        self,
        model_name: str,
        name: str | None = None,
        fmt: Literal["yml", "yaml", "csv"] = "csv",
    ):
        """Generate parameters for a model.

        Parameters
        ----------
        model_name : str
            The model.
        name : str | None
            The name of the parameters.
        fmt : Literal["yml", "yaml", "csv"]
            The parameter format.
        """
        model = self.load_model(model_name)
        name = name if name is not None else model_name + "_parameters"
        self._parameter_registry.generate_parameters(model, name, fmt=fmt)

    def get_parameters_directory(self) -> Path:
        """Get the path to the parameter directory of the project.

        Returns
        -------
        Path
            The path to the project's parameter directory.
        """
        return self._parameter_registry.directory

    @property
    def has_results(self) -> bool:
        """Check if the project has results.

        Returns
        -------
        bool
            Whether the project has results.
        """
        return not self._result_registry.empty

    @property
    def results(self) -> dict[str, str]:
        """Get all project results.

        Returns
        -------
        dict[str, str]
            The results of the project.
        """
        return self._result_registry.items

    def get_result_path(self, name: str) -> Path:
        """Get the path to a result.

        Parameters
        ----------
        name : str
            The name of the result.

        Returns
        -------
        Path
            The path to the result.

        Raises
        ------
        ValueError
            Raised if result does not exist.
        """
        path = self._result_registry.directory / name
        if self._result_registry.is_item(path):
            return path
        else:
            raise ValueError(f"Result '{name}' does not exist.")

    def load_result(self, name: str) -> Result:
        """Load a result.

        Parameters
        ----------
        name : str
            The name of the result.

        Returns
        -------
        Result
            The loaded result.

        Raises
        ------
        ValueError
            Raised if result does not exist.
        """
        try:
            return self._result_registry.load_item(name)
        except ValueError:
            raise ValueError(f"Result '{name}' does not exist.")

    def create_scheme(
        self,
        model: str,
        parameters: str,
        maximum_number_function_evaluations: int | None = None,
        clp_link_tolerance: float = 0.0,
    ) -> Scheme:
        """Create a scheme for optimization.

        Parameters
        ----------
        model : str
            The model to optimize.
        parameters : str
            The initial parameters.
        maximum_number_function_evaluations : int | None
            The maximum number of function evaluations.
        clp_link_tolerance : float
            The CLP link tolerance.

        Returns
        -------
        Scheme
            The created scheme.
        """
        loaded_model = self.load_model(model)
        data = {
            dataset: self.load_data(dataset)
            for dataset in loaded_model.dataset  # type:ignore[attr-defined]
        }
        return Scheme(
            model=loaded_model,
            parameters=self.load_parameters(parameters),
            data=data,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            clp_link_tolerance=clp_link_tolerance,
        )

    def optimize(
        self,
        model: str,
        parameters: str,
        name: str | None = None,
        maximum_number_function_evaluations: int | None = None,
        clp_link_tolerance: float = 0.0,
    ):
        """Optimize a model.

        Parameters
        ----------
        model : str
            The model to optimize.
        parameters : str
            The initial parameters.
        name : str | None
            The name of the result.
        maximum_number_function_evaluations : int | None
            The maximum number of function evaluations.
        clp_link_tolerance : float
            The CLP link tolerance.
        """
        from glotaran.analysis.optimize import optimize

        scheme = self.create_scheme(
            model, parameters, maximum_number_function_evaluations, clp_link_tolerance
        )
        result = optimize(scheme)

        name = name or self._result_registry.create_result_name_for_model(model)
        self._result_registry.save(name, result)

    def markdown(self) -> MarkdownStr:
        """Format the project as a markdown text.

        Returns
        -------
        MarkdownStr : str
            The markdown string.
        """
        md = dedent(
            f"""\
            # Project _{self.folder}_

            pyglotaran version: {self.version}

            ## Data

            {self._data_registry.markdown()}

            ## Model

            {self._model_registry.markdown()}

            ## Parameters

            {self._parameter_registry.markdown()}

            ## Results

            {self._result_registry.markdown()}
            """
        )

        return MarkdownStr(md)
