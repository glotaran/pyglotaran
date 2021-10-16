"""Implementation of the folder Io plugin.

The current implementation is an exact copy of how ``Result.save(path)``
worked in glotaran 0.3.x and meant as an compatibility function.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import save_dataset
from glotaran.io import save_model
from glotaran.io import save_parameters
from glotaran.io import save_result
from glotaran.io import save_scheme
from glotaran.io.interface import ProjectIoInterface
from glotaran.plugin_system.project_io_registration import register_project_io

if TYPE_CHECKING:
    from os import PathLike

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
        result_path: str | PathLike[str],
        format_name: str = None,
        *,
        allow_overwrite: bool = False,
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
            :class:`Result` instance to write.
        result_path : str | PathLike[str]
            Path to write the result data to.
        format_name : str
            Format the result should be saved in, if not provided and it is a file
            it will be inferred from the file extension.
        allow_overwrite : bool
            Whether or not to allow overwriting existing files, by default False
        saving_options : SavingOptions
            Options for the saved result.


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

        return save_result(
            result,
            result_folder / "glotaran_result.yml",
            allow_overwrite=allow_overwrite,
            saving_options=saving_options,
        )


def save_result_to_folder(
    result: Result,
    result_path: str | PathLike[str],
    saving_options: SavingOptions,
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
        :class:`Result` instance to write.
    result_path : str | PathLike[str]
        Path to write the result data to.
    saving_options : SavingOptions
        Options for the saved result.

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
        report_path = result_folder / "result.md"
        report_path.write_text(str(result.markdown()))
        paths.append(report_path.as_posix())

    model_path = result_folder / "model.yml"
    save_model(result.scheme.model, model_path, allow_overwrite=True)
    paths.append(model_path.as_posix())

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

    scheme_path = result_folder / "scheme.yml"
    save_scheme(result.scheme, scheme_path, allow_overwrite=True)
    paths.append(scheme_path.as_posix())

    parameter_history_path = result_folder / "parameter_history.csv"
    result.parameter_history.to_csv(parameter_history_path)
    paths.append(parameter_history_path.as_posix())

    for label, dataset in result.data.items():
        data_path = result_folder / f"{label}.{saving_options.data_format}"
        save_dataset(
            dataset,
            data_path,
            format_name=saving_options.data_format,
            allow_overwrite=True,
            data_filters=saving_options.data_filter,
        )
        paths.append(data_path.as_posix())

    return paths
