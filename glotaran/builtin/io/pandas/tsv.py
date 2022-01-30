"""Module containing TSV io support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.io import ProjectIoInterface
from glotaran.io import load_parameters
from glotaran.io import register_project_io
from glotaran.io import save_parameters

if TYPE_CHECKING:
    from glotaran.parameter import ParameterGroup


@register_project_io(["tsv"])
class TsvProjectIo(ProjectIoInterface):
    """Plugin for TSV data io."""

    def load_parameters(self, file_name: str) -> ParameterGroup:
        """Load parameters from TSV file.

        Parameters
        ----------
        file_name : str
            Name of file to be loaded.

        Returns
        -------
            :class:`ParameterGroup
        """
        return load_parameters(file_name, format_name="csv", sep="\t")

    def save_parameters(
        self,
        parameters: ParameterGroup,
        file_name: str,
        *,
        as_optimized: bool = True,
        replace_infinfinity: bool = True,
    ) -> None:
        """Save a :class:`ParameterGroup` to a TSV file.

        Parameters
        ----------
        parameters : ParameterGroup
            Parameters to be saved to file.
        file_name : str
            File to write the parameters to.
        as_optimized : bool
            Whether to include properties which are the result of optimization.
        replace_infinfinity : bool
            Weather to replace infinity values with empty strings.
        """
        save_parameters(
            parameters,
            file_name,
            format_name="csv",
            sep="\t",
            as_optimized=as_optimized,
            replace_infinfinity=replace_infinfinity,
        )
