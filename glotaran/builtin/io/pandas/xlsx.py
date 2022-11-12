"""Module containing Excel like io support."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.parameter import Parameters
from glotaran.parameter.parameter import OPTION_NAMES_DESERIALIZED
from glotaran.utils.io import safe_dataframe_fillna
from glotaran.utils.io import safe_dataframe_replace


@register_project_io(["xlsx", "ods"])
class ExcelProjectIo(ProjectIoInterface):
    """Plugin for Excel like data io."""

    def load_parameters(self, file_name: str) -> Parameters:
        """Load parameters from XLSX file.

        Parameters
        ----------
        file_name : str
            Name of file to be loaded.

        Returns
        -------
            :class:`Parameters`
        """
        df = pd.read_excel(file_name, na_values=["None", "none"])
        df.columns = [column.lower() for column in df.columns]
        df.rename(columns=OPTION_NAMES_DESERIALIZED, inplace=True)
        safe_dataframe_fillna(df, "minimum", -np.inf)
        safe_dataframe_fillna(df, "maximum", np.inf)
        return Parameters.from_dataframe(df, source=file_name)

    def save_parameters(self, parameters: Parameters, file_name: str):
        """Save a :class:`Parameters` to a Excel file.

        Parameters
        ----------
        parameters : Parameters
            Parameters to be saved to file.
        file_name : str
            File to write the parameters to.
        """
        df = parameters.to_dataframe()
        safe_dataframe_replace(df, "minimum", -np.inf, "")
        safe_dataframe_replace(df, "maximum", np.inf, "")
        df.to_excel(file_name, na_rep="None", index=False)
