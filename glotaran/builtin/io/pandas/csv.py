"""Module containing CSV io support."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.parameter import ParameterGroup
from glotaran.utils.io import safe_dataframe_fillna
from glotaran.utils.io import safe_dataframe_replace


@register_project_io(["csv"])
class CsvProjectIo(ProjectIoInterface):
    """Plugin for CSV data io."""

    def load_parameters(self, file_name: str, sep: str = ",") -> ParameterGroup:
        """Load parameters from CSV file.

        Parameters
        ----------
        file_name : str
            Name of file to be loaded.
        sep: str
            Other separators can be used optionally., by default ','

        Returns
        -------
            :class:`ParameterGroup
        """
        df = pd.read_csv(file_name, skipinitialspace=True, na_values=["None", "none"], sep=sep)
        safe_dataframe_fillna(df, "minimum", -np.inf)
        safe_dataframe_fillna(df, "maximum", np.inf)
        return ParameterGroup.from_dataframe(df, source=file_name)

    def save_parameters(
        self,
        parameters: ParameterGroup,
        file_name: str,
        *,
        sep: str = ",",
        as_optimized: bool = True,
        replace_infinfinity: bool = True,
    ) -> None:
        """Save a :class:`ParameterGroup` to a CSV file.

        Parameters
        ----------
        parameters : ParameterGroup
            Parameters to be saved to file.
        file_name : str
            File to write the parameters to.
        sep: str
            Other separators can be used optionally., by default ','
        as_optimized : bool
            Weather to include properties which are the result of optimization.
        replace_infinfinity : bool
            Weather to replace infinity values with empty strings.
        """
        df = parameters.to_dataframe(as_optimized=as_optimized)
        if replace_infinfinity is True:
            safe_dataframe_replace(df, "minimum", -np.inf, "")
            safe_dataframe_replace(df, "maximum", np.inf, "")
        df.to_csv(file_name, na_rep="None", index=False, sep=sep)
