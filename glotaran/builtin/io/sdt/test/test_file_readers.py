import numpy as np
import pandas as pd
import pytest
import xarray as xr

from glotaran.builtin.io.sdt.sdt_file_reader import SdtDataIo
from glotaran.builtin.io.sdt.test import TEMPORAL_DATA


@pytest.mark.parametrize(
    "test_file_path, result_file_path, index",
    [
        (TEMPORAL_DATA["sdt"], TEMPORAL_DATA["csv"], [1]),
        (TEMPORAL_DATA["sdt"], TEMPORAL_DATA["csv"], None),
    ],
)
@pytest.mark.filterwarnings("ignore:There was no `index`:UserWarning")
def test_read_sdt(test_file_path, result_file_path, index):

    sdt_reader = SdtDataIo("sdt")
    test_dataset = sdt_reader.load_dataset(test_file_path, index=index)
    result_df = pd.read_csv(
        result_file_path, skiprows=1, sep=r"\s+", dtype={"Delay": float, "Data": np.uint16}
    )
    result_df.Delay = result_df.Delay * 1e-9
    result_traces = pd.DataFrame([result_df.Data.values], columns=result_df.Delay)
    assert isinstance(test_dataset, xr.Dataset)
    assert np.all(test_dataset.data.T[0] == result_df.Data.values)

    assert test_dataset.data.T.shape == result_traces.values.shape
    assert np.allclose(test_dataset.time, np.array(result_traces.columns))
