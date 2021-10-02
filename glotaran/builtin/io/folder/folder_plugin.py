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

    # def load_result(self, result_path: str) -> Result:
    #     """Create a Result instance from a result path.

    #     Parameters
    #     ----------
    #     result_path : str
    #         Folder containing the result specs.

    #     Returns
    #     -------
    #     Result
    #         Result instance created from the file.

    #     Raises
    #     ------
    #     FileNotFoundError
    #         When ``path`` does not exist.
    #     """
    #     result_folder = Path(result_path)
    #     if not result_folder.exists():
    #         raise FileNotFoundError(result_path)

    #     result_file = (
    #         result_folder if result_folder.is_file() else result_folder / "glotaran_result.yml"
    #     )

    #     return load_result_file(result_file)

    def save_result(
        self,
        result: Result,
        result_path: str,
        *,
        saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
    ) -> list[str]:
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
        save_model(result.scheme.model, result_folder / result.scheme.model_file)
        paths.append((result_folder / result.scheme.model_file).as_posix())

        result.initial_parameters_file = result.scheme.parameters_file = "initial_parameters.csv"
        save_parameters(result.scheme.parameters, result_folder / result.scheme.parameters_file)
        paths.append((result_folder / result.scheme.parameters_file).as_posix())

        result.optimized_parameters_file = "optimized_parameters.csv"
        save_parameters(
            result.optimized_parameters, result_folder / result.optimized_parameters_file
        )
        paths.append((result_folder / result.optimized_parameters_file).as_posix())

        result.scheme_file = "scheme.yml"
        save_scheme(result.scheme, result_folder / result.scheme_file)
        paths.append((result_folder / result.scheme_file).as_posix())

        result.parameter_history_file = "parameter_history.csv"
        result.parameter_history.to_csv(result_folder / result.parameter_history_file)

        result.data_files = {
            label: f"{label}.{saving_options.data_format}" for label in result.data
        }

        for label, data_file in result.data_files.items():
            save_dataset(result.data[label], result_folder / data_file)
            paths.append((result_folder / data_file).as_posix())

        # result.parameter_history_file = "parameter_history.csv"
        # result.parameter_history.to_csv(result_folder / result.parameter_history_file)

        # result.data_files = {
        #     label: f"{label}.{saving_options.data_format}" for label in result.data
        # }

        # for label, data in result.data.items():
        #     nc_path = os.path.join(result_path, f"{label}.nc")
        #     data.to_netcdf(nc_path, engine="netcdf4")

        return paths
