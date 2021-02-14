import pandas as pd

from glotaran.io import register_io
from glotaran.parameter import ParameterGroup


@register_io(["csv"])
class CsvIo:
    @staticmethod
    def read_parameters(fmt: str, file_name: str) -> ParameterGroup:

        df = pd.read_csv(file_name, skipinitialspace=True, na_values=["None", "none"])
        return ParameterGroup.from_dataframe(df, source=file_name)

    @staticmethod
    def write_parameters(fmt: str, file_name: str, parameters: ParameterGroup):
        """Writes a :class:`ParameterGroup` to a CSV file."""
        parameters.to_dataframe().to_csv(file_name, na_rep="None", index=False)
