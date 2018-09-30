"""
Module containing legacy readers for `glotaran 1.5` formats.
In order to compare results with results produced with `glotaran 1.5`,
file readers for the legacy formats are provided.
"""
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


def FLIM_legacy_to_dataframe(file_path: str, zero_pad: bool=False,
                             traces_only=True) \
        -> Union[Tuple[pd.DataFrame, tuple], Tuple[Dict[str, pd.DataFrame], tuple]]:
    """
    File reader for FLIM data saved with `glotaran 1.5`.

    Parameters
    ----------
    file_path: str
        Path to the legacy file which should be read.

    zero_pad: bool, default False
        Weather or not to zeropad the resulting data (zero padded data are easyer to
        reshape in an intensity map form by simply calling `data.reshape(orig_shape).sum(axis=2)`)

    traces_only: bool, default True
        Flag to either return only the selected pixels time traces as `pd.DataFrame` or a
        dict with keys 'time_traces' and 'intensity_map', which also contains the saved
        intensity map (pixel map with entry equal to the sum over the time trace axis)

    Returns
    -------
    Union[Tuple[pd.DataFrame, Tuple], Tuple[Dict[str, pd.DataFrame], Tuple]]
        pd.DataFrame with selected pixels time traces or dict with pixels time traces and
        intesity map (see `traces_only`) and original shape of the data


    """
    with open(file_path) as FLIM:
        lines = [next(FLIM) for x in range(6)]
        if not 'FLIM Image' == lines[2].rstrip():
            raise TypeError("You are trying to read a file format, "
                            "which is not supported by this reader. "
                            "See the docs for help.")
        x_y_shape = tuple(map(int, lines[3].rstrip().split()))
        orig_shape = (*x_y_shape, int(lines[4].rstrip()))
        nr_of_time_traces = int(lines[5].rstrip())
    time_traces = pd.read_csv(file_path, skiprows=6, nrows=nr_of_time_traces,
                              index_col=0, sep=r"\s+")

    if zero_pad:
        zero_padded = pd.DataFrame(np.zeros((np.prod(x_y_shape), orig_shape[-1]),
                                            dtype=np.int),
                                   columns=time_traces.columns)
        time_traces = zero_padded.add(time_traces)

    intensity_map = pd.read_csv(file_path, skiprows=6 + nr_of_time_traces + 2,
                                sep=r"\s+", names=np.arange(x_y_shape[1]))

    time_traces.columns = pd.to_numeric(time_traces.columns)*1e-9

    if traces_only:
        return time_traces, orig_shape
    else:
        return {"time_traces": time_traces, "intensity_map": intensity_map}, orig_shape
