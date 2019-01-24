import numpy as np
import pandas as pd
import pytest

from ..legacy_readers import FLIM_legacy_to_DataFrame
from . import LEGACY_FILES


@pytest.mark.parametrize("zero_pad", [True, False])
@pytest.mark.parametrize("traces_only", [True, False])
def test_FLIM_legacy_to_df(zero_pad, traces_only):
    time_traces_result = pd.read_csv(LEGACY_FILES["flim_traces"], sep=r"\s+",
                                     index_col=0)
    time_traces_result.columns = pd.to_numeric(time_traces_result.columns)*1e-9
    intensity_map_result = pd.read_csv(LEGACY_FILES["flim_map"], sep=r"\s+",
                                       names=np.arange(64))
    flim_data, orig_shape = FLIM_legacy_to_DataFrame(LEGACY_FILES["flim_file"],
                                                     traces_only=traces_only)

    if traces_only:
        time_traces = flim_data
    else:
        time_traces = flim_data["time_traces"]
        assert np.all(flim_data["intensity_map"] == intensity_map_result)

    assert np.all(time_traces == time_traces_result)

    flim_data, orig_shape = FLIM_legacy_to_DataFrame(LEGACY_FILES["flim_file"],
                                                     traces_only=traces_only,
                                                     zero_pad=zero_pad)
    if traces_only:
        time_traces = flim_data
    else:
        time_traces = flim_data["time_traces"]

    assert orig_shape == (64, 64, 256)
    if zero_pad:
        assert time_traces.shape == (64*64, 256)
    else:
        assert time_traces.shape == (226, 256)


def test_FLIM_legacy_to_df_exception():
    with pytest.raises(TypeError,
                       match="You are trying to read a file format, "
                             "which is not supported by this reader. "
                             "See the docs for help."):
        FLIM_legacy_to_DataFrame(LEGACY_FILES["flim_traces"])
