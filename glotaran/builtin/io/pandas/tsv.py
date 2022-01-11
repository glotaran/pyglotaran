"""Register TSV parameters."""

from __future__ import annotations

import pandas as pd

from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
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
        df = pd.read_csv(file_name, skipinitialspace=True, na_values=["None", "none"], sep="\t")
        return ParameterGroup.from_dataframe(df, source=file_name)

    def save_parameters(self, parameters: ParameterGroup, file_name: str):
        """Save a :class:`ParameterGroup` to a TSV file.

        Parameters
        ----------
        parameters : ParameterGroup
            Parameters to be saved to file.
        file_name : str
            File to write the parameters to.
        """
        parameters.to_dataframe().to_csv(file_name, na_rep="None", index=False, sep="\t")
