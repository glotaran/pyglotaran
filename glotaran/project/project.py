"""The glotaran project module."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from importlib.metadata import distribution
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from warnings import warn

import pandas as pd
import xarray as xr

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.deprecation.deprecation_utils import GlotaranApiDeprecationWarning
from glotaran.deprecation.deprecation_utils import check_overdue
from glotaran.io.prepare_dataset import add_svd_to_dataset
from glotaran.model import Model
from glotaran.parameter import Parameters
from glotaran.project.project_data_registry import ProjectDataRegistry
from glotaran.project.project_model_registry import ProjectModelRegistry
from glotaran.project.project_parameter_registry import ProjectParameterRegistry
from glotaran.project.project_result_registry import ProjectResultRegistry
from glotaran.project.result import Result
from glotaran.project.scheme import Scheme
from glotaran.utils.io import make_path_absolute_if_relative
from glotaran.utils.ipython import MarkdownStr
from glotaran.utils.ipython import display_file

if TYPE_CHECKING:
    from collections.abc import Hashable

    from glotaran.typing.types import LoadableDataset

TEMPLATE = "version: {gta_version}"

PROJECT_FILE_NAME = "project.gta"


@dataclass
class Project:
    """A project represents a projectfolder on disk which contains a project file.

    A project file is a file in `yml` format with name `project.gta`
    """

    version: str = field(init=False)

    file: Path
    folder: Path = field(default=None)  # type:ignore[assignment]

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
            folder, project_file = folder.parent, folder
        else:
            project_file = folder / PROJECT_FILE_NAME

        if project_file.is_file() is False:
            if create_if_not_exist is False:
                raise FileNotFoundError(f"Project file {project_file.as_posix()} does not exists.")
            Project.create(folder)

        project_dict = load_dict(project_file, True)
        project_dict["file"] = project_file
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
    def data(self) -> Mapping[str, Path]:
        """Get all project datasets.

        Returns
        -------
        Mapping[str, Path]
            The models of the datasets.
        """
        return self._data_registry.items

    def load_data(
        self,
        dataset_name: str,
        *,
        add_svd: bool = False,
        lsv_dim: Hashable = "time",
        rsv_dim: Hashable = "spectral",
    ) -> xr.Dataset:
        """Load a dataset, with SVD data if ``add_svd`` is ``True``.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        add_svd: bool
            Whether or not to calculate and add SVD data. Defaults to False.
        lsv_dim: Hashable
            Dimension of the left singular vectors. Defaults to "time".
        rsv_dim: Hashable
            Dimension of the right singular vectors. Defaults to "spectral",

        Returns
        -------
        xr.Dataset
            The loaded dataset, with SVD data if ``add_svd`` is ``True``.

        Raises
        ------
        ValueError
            Raised if the dataset does not exist.


        .. # noqa: DAR402
        """
        dataset = self._data_registry.load_item(dataset_name)
        if isinstance(dataset, xr.DataArray):
            dataset = dataset.to_dataset(name="data")
        if add_svd is True:
            add_svd_to_dataset(dataset, name="data", lsv_dim=lsv_dim, rsv_dim=rsv_dim)
        return dataset

    def import_data(
        self,
        dataset: LoadableDataset | Mapping[str, LoadableDataset],
        dataset_name: str | None = None,
        allow_overwrite: bool = False,
        ignore_existing: bool = True,
    ):
        """Import a dataset by saving it as an .nc file in the project's data folder.

        Parameters
        ----------
        dataset : LoadableDataset
            Dataset instance or path to a dataset.
        dataset_name : str | None
            The name of the dataset (needs to be provided when dataset is an xarray instance).
            Defaults to None.
        allow_overwrite: bool
            Whether to overwrite an existing dataset.
        ignore_existing: bool
            Whether to skip import if the dataset already exists and allow_overwrite is False.
            Defaults to ``True``.
        """
        if not isinstance(dataset, Mapping) or isinstance(dataset, (xr.Dataset, xr.DataArray)):
            dataset = {dataset_name: dataset}

        for key, value in dataset.items():
            self._data_registry.import_data(
                value,
                dataset_name=key,
                allow_overwrite=allow_overwrite,
                ignore_existing=ignore_existing,
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
    def models(self) -> Mapping[str, Path]:
        """Get all project models.

        Returns
        -------
        Mapping[str, Path]
            The models of the project.
        """
        return self._model_registry.items

    def load_model(self, model_name: str) -> Model:
        """Load a model.

        Parameters
        ----------
        model_name : str
            The name of the model.

        Returns
        -------
        Model
            The loaded model.

        Raises
        ------
        ValueError
            Raised if the model does not exist.


        .. # noqa: DAR402
        """
        return self._model_registry.load_item(model_name)

    def show_model_definition(self, model_name: str, syntax: str | None = None) -> MarkdownStr:
        """Show model definition file content with syntax highlighting.

        Parameters
        ----------
        model_name: str
            The name of the model.
        syntax: str | None
            Syntax used for syntax highlighting. Defaults to None which means that the syntax is
            inferred based on the file extension. Pass the value ``""`` to deactivate syntax
            highlighting.

        Returns
        -------
        MarkdownStr
            Model definition file content with syntax highlighting to render in ipython.
        """
        return display_file(self.models[model_name], syntax=syntax)

    def validate(self, model_name: str, parameters_name: str | None = None) -> MarkdownStr:
        """Check that the model is valid, list all issues in the model if there are any.

        If ``parameters_name`` also consider the ``Parameters`` when validating.

        Parameters
        ----------
        model_name: str
            The name of the model to validate.
        parameters_name: str | None
            The name of the parameters to use when validating. Defaults to ``None`` which means
            that parameters are not considered when validating the model.

        Returns
        -------
        MarkdownStr
            Text indicating if the model is valid or not.
        """
        model = self.load_model(model_name)
        return model.validate(
            self.load_parameters(parameters_name) if parameters_name is not None else None
        )

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
        check_overdue("Project.generate_model", "0.8.0")
        warn(
            GlotaranApiDeprecationWarning(
                "Usage of 'Project.generate_model' was deprecated without replacement.\n"
                "This usage will be an error in version: '0.8.0'."
            ),
            stacklevel=2,
        )
        self._model_registry.generate_model(
            model_name,
            generator_name,
            generator_arguments,  # type:ignore[arg-type]
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
    def parameters(self) -> Mapping[str, Path]:
        """Get all project parameters.

        Returns
        -------
        Mapping[str, Path]
            The parameters of the project.
        """
        return self._parameter_registry.items

    def load_parameters(self, parameters_name: str) -> Parameters:
        """Load parameters.

        Parameters
        ----------
        parameters_name : str
            The name of the parameters.

        Raises
        ------
        ValueError
            Raised if parameters do not exist.

        Returns
        -------
        Parameters
            The loaded parameters.


        .. # noqa: D414
        .. # noqa: DAR402
        """
        return self._parameter_registry.load_item(parameters_name)

    def show_parameters_definition(
        self, parameters_name: str, syntax: str | None = None, *, as_dataframe: bool | None = None
    ) -> MarkdownStr | pd.DataFrame:
        """Show parameters definition file content with syntax highlighting.

        Parameters
        ----------
        parameters_name: str
            The name of the parameters.
        syntax: str | None
            Syntax used for syntax highlighting. Defaults to None which means that the syntax is
            inferred based on the file extension. Pass the value ``""`` to deactivate syntax
            highlighting.
        as_dataframe: bool | None
            Whether or not to show the ``Parameters`` definition as pandas.DataFrame (mostly useful
            for non string formats). Defaults to None which means that it will be inferred to
            ``True`` for known non string formats like ``xlsx``.

        Returns
        -------
        MarkdownStr | pd.DataFrame
            Parameters definition file content with syntax highlighting to render in ipython.
        """
        if as_dataframe is True or (
            as_dataframe is None and self.parameters[parameters_name].suffix in [".xlsx", ".ods"]
        ):
            return self.load_parameters(parameters_name).to_dataframe()
        return display_file(self.parameters[parameters_name], syntax=syntax)

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
        check_overdue("Project.generate_parameters", "0.8.0")
        warn(
            GlotaranApiDeprecationWarning(
                "Usage of 'Project.generate_parameters' was deprecated without replacement.\n"
                "This usage will be an error in version: '0.8.0'."
            ),
            stacklevel=2,
        )
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
    def results(self) -> Mapping[str, Path]:
        """Get all project results.

        Returns
        -------
        Mapping[str, Path]
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


        .. # noqa: DAR402
        """
        return self._result_registry._latest_result_path_fallback(result_name, latest=latest)

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


        .. # noqa: DAR402
        """
        return self._result_registry._loader(
            self._result_registry._latest_result_path_fallback(result_name, latest=latest)
        )

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
        data_lookup_override: Mapping[str, LoadableDataset] | None = None,
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
        data_lookup_override: Mapping[str, LoadableDataset] | None
            Allows to bypass the default dataset lookup in the project ``data`` folder and use a
            different dataset for the optimization without changing the model. This is especially
            useful when working with preprocessed data. Defaults to ``None``.

        Returns
        -------
        Scheme
            The created scheme.
        """
        if data_lookup_override is None:
            data_lookup_override = {}
        loaded_model = self.load_model(model_name)
        data_lookup_override = {
            dataset_name: dataset_value
            for dataset_name, dataset_value in data_lookup_override.items()
            if dataset_name in loaded_model.dataset
        }
        data = {
            dataset_name: self.data[dataset_name]
            for dataset_name in loaded_model.dataset
            if dataset_name not in data_lookup_override
        }
        return Scheme(
            model=loaded_model,
            parameters=self.load_parameters(parameters_name),
            data=data | data_lookup_override,
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
        data_lookup_override: Mapping[str, LoadableDataset] | None = None,
    ) -> Result:
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
        data_lookup_override: Mapping[str, LoadableDataset] | None
            Allows to bypass the default dataset lookup in the project ``data`` folder and use a
            different dataset for the optimization without changing the model. This is especially
            useful when working with preprocessed data. Defaults to ``None``.

        Returns
        -------
        Result
            Result of the optimization.
        """
        from glotaran.optimization.optimize import optimize

        scheme = self.create_scheme(
            model_name=model_name,
            parameters_name=parameters_name,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            clp_link_tolerance=clp_link_tolerance,
            data_lookup_override=data_lookup_override,
        )
        result = optimize(scheme)

        result_name = result_name or model_name
        self._result_registry.save(result_name, result)
        return result

    def markdown(self) -> MarkdownStr:
        """Format the project as a markdown text.

        Returns
        -------
        MarkdownStr : str
            The markdown string.
        """
        md = f"""\
            # Project (_{self.folder.name}_)

            pyglotaran version: `{self.version}`

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
        """Create a markdown representation.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str :
            The markdown representation as string.
        """
        return str(self.markdown())
