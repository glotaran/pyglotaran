"""Implementation of the folder Io plugin.

The current implementation is an exact copy of how ``Result.save(path)``
worked in glotaran 0.3.x and meant as an compatibility function.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.io import load_result_file
from glotaran.io import save_dataset
from glotaran.io import save_model
from glotaran.io import save_parameters
from glotaran.io import save_result_file
from glotaran.io import save_scheme
from glotaran.io.interface import ProjectIoInterface
from glotaran.plugin_system.project_io_registration import register_project_io

if TYPE_CHECKING:
    from glotaran.io import SavingOptions
    from glotaran.project import Result


@register_project_io(["folder", "legacy"])
class FolderProjectIo(ProjectIoInterface):
    """Project Io plugin to save result data to a folder.

    There won't be a serialization of the Result object, but simply
    a markdown summary output and the important data saved to files.
    """

    def load_result(self, path: str) -> Result:
        """Create a Result instance from a result path.

        Parameters
        ----------
        path : str
            Folder containing the result specs.

        Returns
        -------
        Result
            Result instance created from the file.

        Raises
        ------
        FileNotFoundError
            When ``path`` does not exist.
        """
        result_folder = Path(path)
        if not result_folder.exists():
            raise FileNotFoundError(path)

        result_file = (
            result_folder if result_folder.is_file() else result_folder / "glotaran_result.yml"
        )

        result = load_result_file(result_file)

        return result

    def save_result(
        self, result: Result, folder: str, saving_options: SavingOptions, allow_overwrite: bool
    ):
        """Save the result to a given folder.

        Returns a list with paths of all saved items.
        The following files are saved:
        * `result.md`: The result with the model formatted as markdown text.
        * `optimized_parameters.csv`: The optimized parameter as csv file.
        * `{dataset_label}.nc`: The result data for each dataset as NetCDF file.

        Parameters
        ----------
        result : Result
            Result instance to be saved.
        folder : str
            The path to the folder in which to save the result.
        saving_options : SavingOptions
            Options for saving the the result.
        allow_overwrite : bool
            Whether or not to allow overwriting existing files, by default False

        Raises
        ------
        ValueError
            If ``folder`` is a file.
        FileExistsError
            If ``folder`` is exists and ``allow_overwrite`` is ``False``.
        """
        result_folder = Path(folder)
        if not result_folder.exists():
            result_folder.mkdir()
        elif result_folder.is_file():
            raise ValueError(f"The path '{result_folder}' is not a directory.")
        elif not allow_overwrite:
            raise FileExistsError

        if saving_options.report:
            report_file = result_folder / "result.md"
            with open(report_file, "w") as f:
                f.write(str(result.markdown()))

        result.scheme.model_file = "model.yml"
        save_model(result.scheme.model, result_folder / result.scheme.model_file)
        result.scheme.parameters_file = "initial_parameters.csv"
        result.initial_parameters_file = result.scheme.parameters_file
        save_parameters(result.scheme.parameters, result_folder / result.scheme.parameters_file)
        result.optimized_parameters_file = "optimized_parameters.csv"
        save_parameters(
            result.optimized_parameters, result_folder / result.optimized_parameters_file
        )
        result.scheme_file = "scheme.yml"
        save_scheme(result.scheme, result_folder / result.scheme_file)

        result.parameter_history_file = "parameter_history.csv"
        result.parameter_history.to_csv(result_folder / result.parameter_history_file)

        result.data_files = {
            label: f"{label}.{saving_options.data_format}" for label in result.data
        }

        for label, data_file in result.data_files.items():
            save_dataset(result.data[label], result_folder / data_file)

        save_result_file(result, result_folder / "glotaran_result.yml")
