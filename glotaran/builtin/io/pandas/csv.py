from __future__ import annotations

import numpy as np
import pandas as pd

from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.parameter import ParameterGroup


@register_project_io(["csv"])
class CsvProjectIo(ProjectIoInterface):
    def load_parameters(self, file_name: str) -> ParameterGroup:
        df = pd.read_csv(file_name, skipinitialspace=True, na_values=["None", "none"])
        df["minimum"].fillna(-np.inf, inplace=True)
        df["maximum"].fillna(np.inf, inplace=True)
        return ParameterGroup.from_dataframe(df, source=file_name)

    def save_parameters(self, parameters: ParameterGroup, file_name: str, *, sep=","):
        """Save a :class:`ParameterGroup` to a CSV file."""
        df = parameters.to_dataframe()
        df["minimum"].replace([-np.inf], "", inplace=True)
        df["maximum"].replace([np.inf], "", inplace=True)
        df.to_csv(file_name, na_rep="None", index=False, sep=sep)
