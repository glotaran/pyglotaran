"""Module containing Excel like io support."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.parameter import Parameters
from glotaran.parameter.parameter import OPTION_NAMES_DESERIALIZED
from glotaran.utils.io import normalize_dataframe_columns
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
        parameter_df = (
            pd.read_excel(file_name, na_values=["None", "none"])
            .pipe(normalize_dataframe_columns, rename_dict=OPTION_NAMES_DESERIALIZED)
            .pipe(safe_dataframe_fillna, column_name="minimum", fill_value=-np.inf)
            .pipe(safe_dataframe_fillna, column_name="maximum", fill_value=np.inf)
        )
        return Parameters.from_dataframe(parameter_df, source=file_name)

    def save_parameters(self, parameters: Parameters, file_name: str) -> None:
        """Save a :class:`Parameters` to a Excel file.

        Parameters
        ----------
        parameters : Parameters
            Parameters to be saved to file.
        file_name : str
            File to write the parameters to.
        """
        parameter_df = (
            parameters.to_dataframe()
            .pipe(
                safe_dataframe_replace,
                column_name="minimum",
                to_be_replaced_values=-np.inf,
                replace_value="",
            )
            .pipe(
                safe_dataframe_replace,
                column_name="maximum",
                to_be_replaced_values=np.inf,
                replace_value="",
            )
        )
        parameter_df.to_excel(file_name, na_rep="None", index=False)
