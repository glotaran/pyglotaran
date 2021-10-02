"""Implementation of the folder Io plugin.

The current implementation is an exact copy of how ``Result.save(path)``
worked in glotaran 0.3.x and meant as an compatibility function.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.io import save_dataset
from glotaran.io import save_model
from glotaran.io import save_parameters
from glotaran.io import save_scheme
from glotaran.io.interface import ProjectIoInterface
from glotaran.plugin_system.project_io_registration import SAVING_OPTIONS_DEFAULT
from glotaran.plugin_system.project_io_registration import register_project_io

if TYPE_CHECKING:
    from glotaran.plugin_system.project_io_registration import SavingOptions
    from glotaran.project import Result


@register_project_io(["folder", "legacy"])
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
    ) -> list[str]:
        """Save the result to a given folder.

        Returns a list with paths of all saved items.
        The following files are saved if not configured otherwise:
        * `result.md`: The result with the model formatted as markdown text.
        * `model.yml`: Model spec file.
        * `scheme.yml`: Scheme spec file.
        * `initial_parameters.csv`: Initially used parameters.
        * `optimized_parameters.csv`: The optimized parameter as csv file.
        * `parameter_history.csv`: Parameter changes over the optimization
        * `{dataset_label}.nc`: The result data for each dataset as NetCDF file.

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


        Returns
        -------
        list[str]
            List of file paths which were created.

        Raises
        ------
        ValueError
            If ``result_path`` is a file.
        """
        result_folder = Path(result_path)
        if result_folder.is_file():
            raise ValueError(f"The path '{result_folder}' is not a directory.")
        result_folder.mkdir(parents=True, exist_ok=True)

        paths = []
        if saving_options.report:
            report_file = result_folder / "result.md"
            report_file.write_text(str(result.markdown()))
            paths.append(report_file.as_posix())

        result.scheme.model_file = "model.yml"
        save_model(
            result.scheme.model, result_folder / result.scheme.model_file, allow_overwrite=True
        )
        paths.append((result_folder / result.scheme.model_file).as_posix())

        result.initial_parameters_file = (
            result.scheme.parameters_file
        ) = f"initial_parameters.{saving_options.parameter_format}"
        save_parameters(
            result.scheme.parameters,
            result_folder / result.scheme.parameters_file,
            format_name=saving_options.parameter_format,
            allow_overwrite=True,
        )
        paths.append((result_folder / result.scheme.parameters_file).as_posix())

        result.optimized_parameters_file = (
            f"optimized_parameters.{saving_options.parameter_format}"
        )
        save_parameters(
            result.optimized_parameters,
            result_folder / result.optimized_parameters_file,
            format_name=saving_options.parameter_format,
            allow_overwrite=True,
        )
        paths.append((result_folder / result.optimized_parameters_file).as_posix())

        result.scheme_file = "scheme.yml"
        save_scheme(result.scheme, result_folder / result.scheme_file, allow_overwrite=True)
        paths.append((result_folder / result.scheme_file).as_posix())

        result.parameter_history_file = "parameter_history.csv"
        result.parameter_history.to_csv(result_folder / result.parameter_history_file)
        paths.append((result_folder / result.parameter_history_file).as_posix())

        result.data_files = {
            label: f"{label}.{saving_options.data_format}" for label in result.data
        }

        for label, data_file in result.data_files.items():
            save_dataset(
                result.data[label],
                result_folder / data_file,
                format_name=saving_options.data_format,
                allow_overwrite=True,
            )
            paths.append((result_folder / data_file).as_posix())

        return paths
