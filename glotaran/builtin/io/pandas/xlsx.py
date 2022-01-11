"""Register XLSX parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.parameter import ParameterGroup
from glotaran.utils.io import safe_parameters_fillna
from glotaran.utils.io import safe_parameters_replace


@register_project_io(["xlsx", "ods"])
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
        safe_parameters_fillna(df, "minimum", -np.inf)
        safe_parameters_fillna(df, "maximum", np.inf)
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
        safe_parameters_replace(df, "minimum", -np.inf, "")
        safe_parameters_replace(df, "maximum", np.inf, "")
        df.to_excel(file_name, na_rep="None", index=False)
