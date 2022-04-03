"""The glotaran project module."""
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from importlib.metadata import distribution
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

TEMPLATE = "version: {gta_version}"

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
        self.version = distribution("pyglotaran").version

        self._data_registry = ProjectDataRegistry(self.folder)
        self._model_registry = ProjectModelRegistry(self.folder)
        self._parameter_registry = ProjectParameterRegistry(self.folder)
        self._result_registry = ProjectResultRegistry(self.folder)

    @staticmethod
    def create(folder: str | Path, allow_overwrite: bool = False) -> Project:
        """Create a new project folder and file.

        Parameters
        ----------
        folder : str | Path | None
            The folder where the project will be created. If ``None``, the current work
            directory will be used.
        allow_overwrite: bool
            Whether to overwrite an existing project file.

        Returns
        -------
        Project
            The created project.

        Raises
        ------
        FileExistsError
            Raised if the project file already exists and `allow_overwrite=False`.
        """
        project_folder = make_path_absolute_if_relative(Path(folder))
        project_folder.mkdir(parents=True, exist_ok=True)
        project_file = project_folder / PROJECT_FILE_NAME
        if project_file.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Project file '{project_file}' already exist. "
                "Set `allow_overwrite=True` to overwrite."
            )
        project_file.write_text(TEMPLATE.format(gta_version=distribution("pyglotaran").version))
        return Project.open(project_file, create_if_not_exist=True)

    @classmethod
    def open(cls, project_folder_or_file: str | Path, create_if_not_exist: bool = True) -> Project:
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
            The project instance.

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

        if file.is_file() is False:
            if create_if_not_exist is False:
                raise FileNotFoundError(f"Project file {file.as_posix()} does not exists.")
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
    def data(self) -> dict[str, Path]:
        """Get all project datasets.

        Returns
        -------
        dict[str, Path]
            The models of the datasets.
        """
        return self._data_registry.items

    def load_data(self, dataset_name: str) -> xr.Dataset | xr.DataArray:
        """Load a dataset.

        Parameters
        ----------
        dataset_name : str
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
            return self._data_registry.load_item(dataset_name)
        except ValueError as e:
            raise ValueError(f"Dataset {dataset_name!r} does not exist.") from e

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
        self._data_registry.import_data(
            path, name=name, allow_overwrite=allow_overwrite, ignore_existing=ignore_existing
        )

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
    def models(self) -> dict[str, Path]:
        """Get all project models.

        Returns
        -------
        dict[str, Path]
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
        except ValueError as e:
            raise ValueError(f"Model {name!r} does not exist.") from e

    def generate_model(
        self,
        model_name: str,
        generator_name: str,
        generator_arguments: dict[str, Any],
        *,
        allow_overwrite: bool = False,
        ignore_existing: bool = False,
    ):
        """Generate a model.

        Parameters
        ----------
        model_name : str
            The name of the model.
        generator_name : str
            The generator for the model.
        generator_arguments : dict[str, Any]
            Arguments for the generator.
        allow_overwrite: bool
            Whether to overwrite an existing model.
        ignore_existing: bool
            Whether to ignore generation of a model file if it already exists.
        """
        self._model_registry.generate_model(
            model_name,
            generator_name,
            generator_arguments,
            allow_overwrite=allow_overwrite,
            ignore_existing=ignore_existing,
        )

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
    def parameters(self) -> dict[str, Path]:
        """Get all project parameters.

        Returns
        -------
        dict[str, Path]
            The parameters of the project.
        """
        return self._parameter_registry.items

    def load_parameters(self, parameters_name: str) -> ParameterGroup:
        """Load parameters.

        Parameters
        ----------
        parameters_name : str
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
            return self._parameter_registry.load_item(parameters_name)
        except ValueError as e:
            raise ValueError(f"Parameters '{parameters_name}' does not exist.") from e

    def generate_parameters(
        self,
        model_name: str,
        parameters_name: str | None = None,
        *,
        format_name: Literal["yml", "yaml", "csv"] = "csv",
        allow_overwrite: bool = False,
        ignore_existing: bool = False,
    ):
        """Generate parameters for a model.

        Parameters
        ----------
        model_name : str
            The model.
        parameters_name : str | None
            The name of the parameters. If ``None`` it will be <model_name>_parameters.
        format_name : Literal["yml", "yaml", "csv"]
            The parameter format.
        allow_overwrite: bool
            Whether to overwrite existing parameters.
        ignore_existing: bool
            Whether to ignore generation of a parameter file if it already exists.
        """
        model = self.load_model(model_name)
        parameters_name = (
            parameters_name if parameters_name is not None else f"{model_name}_parameters"
        )
        self._parameter_registry.generate_parameters(
            model,
            parameters_name,
            format_name=format_name,
            allow_overwrite=allow_overwrite,
            ignore_existing=ignore_existing,
        )

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
    def results(self) -> dict[str, Path]:
        """Get all project results.

        Returns
        -------
        dict[str, Path]
            The results of the project.
        """
        return self._result_registry.items

    def get_result_path(self, result_name: str, *, latest: bool = False) -> Path:
        """Get the path to a result with name ``name``.

        Parameters
        ----------
        result_name : str
            The name of the result.
        latest: bool
            Flag to deactivate warning about using latest result. Defaults to False

        Returns
        -------
        Path
            The path to the result.

        Raises
        ------
        ValueError
            Raised if result does not exist.
        """
        result_name = self._result_registry._latest_result_name_fallback(
            result_name, latest=latest
        )

        path = self._result_registry.directory / result_name
        if self._result_registry.is_item(path):
            return path

        raise ValueError(f"Result {result_name!r} does not exist.")

    def get_latest_result_path(self, result_name: str) -> Path:
        """Get the path to a result with name ``name``.

        Parameters
        ----------
        result_name : str
            The name of the result.

        Returns
        -------
        Path
            The path to the result.

        Raises
        ------
        ValueError
            Raised if result does not exist.


        .. # noqa: DAR402
        """
        result_name = re.sub(self._result_registry.result_pattern, "", result_name)
        return self.get_result_path(result_name, latest=True)

    def load_result(self, result_name: str, *, latest: bool = False) -> Result:
        """Load a result.

        Parameters
        ----------
        result_name : str
            The name of the result.
        latest: bool
            Flag to deactivate warning about using latest result. Defaults to False

        Returns
        -------
        Result
            The loaded result.

        Raises
        ------
        ValueError
            Raised if result does not exist.
        """
        result_name = self._result_registry._latest_result_name_fallback(
            result_name, latest=latest
        )
        try:
            return self._result_registry.load_item(result_name)
        except ValueError as e:
            raise ValueError(f"Result {result_name!r} does not exist.") from e

    def load_latest_result(self, result_name: str) -> Result:
        """Load a result.

        Parameters
        ----------
        result_name : str
            The name of the result.

        Returns
        -------
        Result
            The loaded result.

        Raises
        ------
        ValueError
            Raised if result does not exist.


        .. # noqa: DAR402
        """
        result_name = re.sub(self._result_registry.result_pattern, "", result_name)
        return self.load_result(result_name, latest=True)

    def create_scheme(
        self,
        model_name: str,
        parameters_name: str,
        maximum_number_function_evaluations: int | None = None,
        clp_link_tolerance: float = 0.0,
    ) -> Scheme:
        """Create a scheme for optimization.

        Parameters
        ----------
        model_name : str
            The model to optimize.
        parameters_name : str
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
        loaded_model = self.load_model(model_name)
        data = {
            dataset: self.load_data(dataset)
            for dataset in loaded_model.dataset  # type:ignore[attr-defined]
        }
        return Scheme(
            model=loaded_model,
            parameters=self.load_parameters(parameters_name),
            data=data,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            clp_link_tolerance=clp_link_tolerance,
        )

    def optimize(
        self,
        model_name: str,
        parameters_name: str,
        result_name: str | None = None,
        maximum_number_function_evaluations: int | None = None,
        clp_link_tolerance: float = 0.0,
    ):
        """Optimize a model.

        Parameters
        ----------
        model_name : str
            The model to optimize.
        parameters_name : str
            The initial parameters.
        result_name : str | None
            The name of the result.
        maximum_number_function_evaluations : int | None
            The maximum number of function evaluations.
        clp_link_tolerance : float
            The CLP link tolerance.
        """
        from glotaran.optimization.optimize import optimize

        scheme = self.create_scheme(
            model_name, parameters_name, maximum_number_function_evaluations, clp_link_tolerance
        )
        result = optimize(scheme)

        result_name = result_name or model_name
        self._result_registry.save(result_name, result)

    def markdown(self) -> MarkdownStr:
        """Format the project as a markdown text.

        Returns
        -------
        MarkdownStr : str
            The markdown string.
        """
        folder_as_posix = self.folder.as_posix()  # type:ignore[union-attr]
        md = f"""\
            # Project _{folder_as_posix}_

            pyglotaran version: {self.version}

            ## Data

            {self._data_registry.markdown(join_indentation=12)}

            ## Model

            {self._model_registry.markdown(join_indentation=12)}

            ## Parameters

            {self._parameter_registry.markdown(join_indentation=12)}

            ## Results

            {self._result_registry.markdown(join_indentation=12)}
            """

        return MarkdownStr(dedent(md))

    def _repr_markdown_(self) -> str:
        """Create a markdown respresentation.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str :
            The markdown representation as string.
        """
        return str(self.markdown())
