"""Module containing TSV io support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.io import ProjectIoInterface
from glotaran.io import load_parameters
from glotaran.io import register_project_io
from glotaran.io import save_parameters

if TYPE_CHECKING:
    from glotaran.parameter import Parameters


@register_project_io(["tsv"])
class TsvProjectIo(ProjectIoInterface):
    """Plugin for TSV data io."""

    def load_parameters(self, file_name: str) -> Parameters:
        """Load parameters from TSV file.

        Parameters
        ----------
        file_name : str
            Name of file to be loaded.

        Returns
        -------
            :class:`Parameters`
        """
        return load_parameters(file_name, format_name="csv", sep="\t")

    def save_parameters(
        self,
        parameters: Parameters,
        file_name: str,
        *,
        replace_infinfinity: bool = True,
    ) -> None:
        """Save a :class:`Parameters` to a TSV file.

        Parameters
        ----------
        parameters : Parameters
            Parameters to be saved to file.
        file_name : str
            File to write the parameters to.
        replace_infinfinity : bool
            Weather to replace infinity values with empty strings.
        """
        save_parameters(
            parameters,
            file_name,
            format_name="csv",
            sep="\t",
            replace_infinfinity=replace_infinfinity,
        )
