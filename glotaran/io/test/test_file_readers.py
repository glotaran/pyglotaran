import numpy as np
import pandas as pd
import xarray as xr
import pytest

from glotaran.io.sdt_file_reader import read_sdt
from ..legacy_readers import FLIM_legacy_to_DataFrame
from . import TEMPORAL_DATA, FLIM_DATA


@pytest.mark.parametrize("flim, test_file_path, result_file_path, index", [
    (False, TEMPORAL_DATA["sdt"], TEMPORAL_DATA["csv"], [1]),
    (True, FLIM_DATA["sdt"], FLIM_DATA["csv"], None)
])
def test_read_sdt(flim, test_file_path, result_file_path, index):

    test_dataset = read_sdt(file_path=test_file_path, index=index,
                            flim=flim)
    if flim:
        result_dict, orig_shape = FLIM_legacy_to_DataFrame(FLIM_DATA["csv"],
                                                           traces_only=False, zero_pad=True)
        result_intensity_map = result_dict["intensity_map"]
        result_traces = result_dict["time_traces"]
        assert isinstance(test_dataset, xr.Dataset)
        assert np.all(test_dataset.data_intensity_map == result_intensity_map)
        assert test_dataset.full_data.shape == orig_shape
    else:
        result_df = pd.read_csv(result_file_path, skiprows=1, sep=r"\s+",
                                dtype={"Delay": np.float, "Data": np.uint16})
        result_df.Delay = result_df.Delay * 1e-9
        result_traces = pd.DataFrame([result_df.Data.values], columns=result_df.Delay)
        assert isinstance(test_dataset, xr.Dataset)
        assert np.all(test_dataset.data.T[0] == result_df.Data.values)

    assert test_dataset.data.T.shape == result_traces.values.shape
    assert np.allclose(test_dataset.time, np.array(result_traces.columns))
