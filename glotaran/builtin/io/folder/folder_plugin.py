"""Implementation of the folder Io plugin.

The current implementation is an exact copy of how ``Result.save(path)``
worked in glotaran 0.3.x and meant as an compatibility function.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from glotaran.io.interface import ProjectIoInterface
from glotaran.plugin_system.project_io_registration import register_project_io

if TYPE_CHECKING:
    from glotaran.project import Result


@register_project_io(["folder", "legacy"])
class FolderProjectIo(ProjectIoInterface):
    """Project Io plugin to save result data to a folder.

    There won't be a serialization of the Result object, but simply
    a markdown summary output and the important data saved to files.
    """

    def save_result(self, result_path: str, result: Result) -> list[str]:
        """Save the result to a given folder.

        Returns a list with paths of all saved items.
        The following files are saved:
        * `result.md`: The result with the model formatted as markdown text.
        * `optimized_parameters.csv`: The optimized parameter as csv file.
        * `{dataset_label}.nc`: The result data for each dataset as NetCDF file.

        Parameters
        ----------
        result_path : str
            The path to the folder in which to save the result.
        result : Result
            Result instance to be saved.

        Returns
        -------
        list[str]
            List of file paths which were created.

        Raises
        ------
        OSError
            If the folder to save the result in doesn't exist or couldn't be created.
        """
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        elif not os.path.isdir(result_path):
            raise OSError(f"The path '{result_path}' is not a directory.")

        paths = []

        md_path = os.path.join(result_path, "result.md")
        with open(md_path, "w") as f:
            f.write(result.markdown())
        paths.append(md_path)

        csv_path = os.path.join(result_path, "optimized_parameters.csv")
        result.optimized_parameters.to_csv(csv_path)
        paths.append(csv_path)

        for label, data in result.data.items():
            nc_path = os.path.join(result_path, f"{label}.nc")
            data.to_netcdf(nc_path, engine="netcdf4")
            paths.append(nc_path)

        return paths
