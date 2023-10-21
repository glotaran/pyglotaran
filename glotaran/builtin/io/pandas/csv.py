"""Module containing CSV io support."""

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


@register_project_io(["csv"])
class CsvProjectIo(ProjectIoInterface):
    """Plugin for CSV data io."""

    def load_parameters(self, file_name: str, sep: str = ",") -> Parameters:
        """Load parameters from CSV file.

        Parameters
        ----------
        file_name : str
            Name of file to be loaded.
        sep: str
            Other separators can be used optionally., by default ','

        Returns
        -------
            :class:`Parameters
        """
        parameter_df = (
            pd.read_csv(file_name, skipinitialspace=True, na_values=["None", "none"], sep=sep)
            .pipe(normalize_dataframe_columns, rename_dict=OPTION_NAMES_DESERIALIZED)
            .pipe(safe_dataframe_fillna, column_name="minimum", fill_value=-np.inf)
            .pipe(safe_dataframe_fillna, column_name="maximum", fill_value=np.inf)
        )
        return Parameters.from_dataframe(parameter_df, source=file_name)

    def save_parameters(
        self,
        parameters: Parameters,
        file_name: str,
        *,
        sep: str = ",",
        as_optimized: bool = True,
        replace_infinity: bool = True,
    ) -> None:
        """Save a :class:`Parameters` to a CSV file.

        Parameters
        ----------
        parameters : Parameters
            Parameters to be saved to file.
        file_name : str
            File to write the parameters to.
        sep: str
            Other separators can be used optionally., by default ','
        as_optimized : bool
            Weather to include properties which are the result of optimization.
        replace_infinity : bool
            Weather to replace infinity values with empty strings.
        """
        parameter_df = parameters.to_dataframe()
        if replace_infinity is True:
            parameter_df = parameter_df.pipe(
                safe_dataframe_replace,
                column_name="minimum",
                to_be_replaced_values=-np.inf,
                replace_value="",
            ).pipe(
                safe_dataframe_replace,
                column_name="maximum",
                to_be_replaced_values=np.inf,
                replace_value="",
            )
        parameter_df.to_csv(file_name, na_rep="None", index=False, sep=sep)
