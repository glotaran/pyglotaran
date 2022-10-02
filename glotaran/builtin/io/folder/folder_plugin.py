"""Implementation of the folder Io plugin.

The current implementation is an exact copy of how ``Result.save(path)``
worked in glotaran 0.3.x and meant as an compatibility function.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from glotaran.deprecation import warn_deprecated
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.io import save_result
from glotaran.io.interface import ProjectIoInterface
from glotaran.plugin_system.project_io_registration import SAVING_OPTIONS_DEFAULT
from glotaran.plugin_system.project_io_registration import register_project_io

if TYPE_CHECKING:
    from glotaran.plugin_system.project_io_registration import SavingOptions
    from glotaran.project import Result


@register_project_io("legacy")
class LegacyProjectIo(ProjectIoInterface):
    """Project Io plugin to save result data in a backward compatible manner."""

    def save_result(
        self,
        result: Result,
        result_path: str,
        *,
        saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
    ) -> list[str]:
        """Save the result to a given folder.

        Warning
        -------
        Deprecated use ``glotaran.io.save_result(result, Path(result_path) / 'result.yml')``
        instead.

        Returns a list with paths of all saved items.
        The following files are saved if not configured otherwise:
        * ``result.md``: The result with the model formatted as markdown text.
        * ``result.yml``: Yaml spec file of the result
        * ``model.yml``: Model spec file.
        * ``scheme.yml``: Scheme spec file.
        * ``initial_parameters.csv``: Initially used parameters.
        * ``optimized_parameters.csv``: The optimized parameter as csv file.
        * ``parameter_history.csv``: Parameter changes over the optimization
        * ``{dataset_label}.nc``: The result data for each dataset as NetCDF file.

        Parameters
        ----------
        result : Result
            Result instance to be saved.
        result_path : str
            The path to the folder in which to save the result.
        saving_options : SavingOptions
            Options for saving the the result.

        Returns
        -------
        list[str]
            List of file paths which were created.
        """
        warn_deprecated(
            deprecated_qual_name_usage=(
                "glotaran.io.save_result(result, result_path, format_name='legacy')"
            ),
            new_qual_name_usage=(
                "glotaran.io.save_result(result, Path(result_path) / 'result.yml')"
            ),
            to_be_removed_in_version="0.8.0",
            stacklevel=5,
        )

        return save_result(
            result=result,
            result_path=Path(result_path) / "result.yml",
            saving_options=saving_options,
            allow_overwrite=True,
        )


@register_project_io("folder")
class FolderProjectIo(ProjectIoInterface):
    """Project Io plugin to save result data to a folder.

    There won't be a serialization of the Result object, but simply
    a markdown summary output and the important data saved to files.
    """

    def save_result(
        self,
        result: Result,
        result_path: str,
        *,
        saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
        used_inside_of_plugin: bool = False,
    ) -> list[str]:
        """Save the result to a given folder.

        Returns a list with paths of all saved items.
        The following files are saved if not configured otherwise:
        * ``result.md``: The result with the model formatted as markdown text.
        * ``initial_parameters.csv``: Initially used parameters.
        * ``optimized_parameters.csv``: The optimized parameter as csv file.
        * ``parameter_history.csv``: Parameter changes over the optimization
        * ``{dataset_label}.nc``: The result data for each dataset as NetCDF file.

        Note
        ----
        As a side effect it populates the file path properties of ``result`` which can be
        used in other plugins (e.g. the ``yml`` save_result).

        Parameters
        ----------
        result : Result
            Result instance to be saved.
        result_path : str
            The path to the folder in which to save the result.
        saving_options : SavingOptions
            Options for saving the the result.
        used_inside_of_plugin: bool
            Denote that this plugin is used from inside another plugin,
            if false a user warning will be thrown. , by default False

        Returns
        -------
        list[str]
            List of file paths which were created.

        Raises
        ------
        ValueError
            If ``result_path`` is a file.
        """
        if used_inside_of_plugin is not True:
            warn(
                UserWarning(
                    "The folder plugin is only intended for internal use by other plugins "
                    "as quick way to save most of the files. The saved result will be incomplete, "
                    "thus it is not recommended to be used directly."
                ),
                stacklevel=4,
            )

        result_folder = Path(result_path)
        if result_folder.is_file():
            raise ValueError(f"The path '{result_folder}' is not a directory.")
        result_folder.mkdir(parents=True, exist_ok=True)

        paths = []
        if saving_options.report:
            report_path = result_folder / "result.md"
            report_path.write_text(str(result.markdown()))
            paths.append(report_path.as_posix())

        initial_parameters_path = f"initial_parameters.{saving_options.parameter_format}"
        save_parameters(
            result.scheme.parameters,
            result_folder / initial_parameters_path,
            format_name=saving_options.parameter_format,
            allow_overwrite=True,
        )
        paths.append((result_folder / initial_parameters_path).as_posix())

        optimized_parameters_path = f"optimized_parameters.{saving_options.parameter_format}"
        save_parameters(
            result.optimized_parameters,
            result_folder / optimized_parameters_path,
            format_name=saving_options.parameter_format,
            allow_overwrite=True,
        )
        paths.append((result_folder / optimized_parameters_path).as_posix())

        parameter_history_path = result_folder / "parameter_history.csv"
        result.parameter_history.to_csv(parameter_history_path)
        paths.append(parameter_history_path.as_posix())

        optimization_history_path = result_folder / "optimization_history.csv"
        result.optimization_history.to_csv(optimization_history_path)
        paths.append(optimization_history_path.as_posix())

        for label, dataset in result.data.items():
            data_path = result_folder / f"{label}.{saving_options.data_format}"
            if saving_options.data_filter is not None:
                dataset = dataset[saving_options.data_filter]
            save_dataset(
                dataset,
                data_path,
                format_name=saving_options.data_format,
                allow_overwrite=True,
            )
            paths.append(data_path.as_posix())

        return paths
