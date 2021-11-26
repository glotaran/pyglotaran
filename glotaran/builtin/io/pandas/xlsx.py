"""Register XLSX parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.parameter import ParameterGroup


@register_project_io(["xlsx"])
class ExcelProjectIo(ProjectIoInterface):
    """Plugin for XLSX data io."""

    def load_parameters(self, file_name: str) -> ParameterGroup:
        """Load parameters from XLSX file.

        Parameters
        ----------
        file_name : str
            Name of file to be loaded.

        Returns
        -------
            :class:`ParameterGroup
        """
        df = pd.read_excel(file_name, na_values=["None", "none"])
        df["minimum"].fillna(-np.inf, inplace=True)
        df["maximum"].fillna(np.inf, inplace=True)
        return ParameterGroup.from_dataframe(df, source=file_name)

    def save_parameters(self, parameters: ParameterGroup, file_name: str):
        """Save a :class:`ParameterGroup` to a Excel file.

        Parameters
        ----------
        parameters : ParameterGroup
            Parameters to be saved to file.
        file_name : str
            File to write the parameters to.
        """
        df = parameters.to_dataframe()
        df["minimum"].replace([-np.inf], "", inplace=True)
        df["maximum"].replace([np.inf], "", inplace=True)
        df.to_excel(file_name, na_rep="None", index=False)
