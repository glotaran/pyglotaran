import numpy as np
import pandas as pd
import pytest

from glotaran.data.external_file_readers import sdt_reader
from ..file_readers import (
    sdt_to_DataFrame,
    DataFrame_to_SpectralTemporalDataset,
    DataFrame_to_FLIMDataset,
    read_sdt,
)
from ..legacy_readers import FLIM_legacy_to_DataFrame
from ..mapper import get_pixel_map
from glotaran.model.dataset import DimensionalityError
from glotaran.data.datasets.specialized_datasets import FLIMDataset
from glotaran.data.datasets.spectral_temporal_dataset import SpectralTemporalDataset
from . import TEMPORAL_DATA, FLIM_DATA


def test_sdt_to_df__temporal():
    result_df = pd.read_csv(TEMPORAL_DATA["csv"], skiprows=1, sep=r"\s+",
                            dtype={"Delay": np.float, "Data": np.uint16})
    result_df.Delay = result_df.Delay * 1e-9
    test_df, orig_shape = sdt_to_DataFrame(TEMPORAL_DATA["sdt"], index=[1])
    assert np.allclose(test_df.columns, result_df.Delay.values)
    assert np.all(test_df.values[0] == result_df.Data.values)
    assert orig_shape == (1, 4096)


def test_sdt_to_df__errors_and_warnings(monkeypatch):
    with pytest.warns(UserWarning, match="There was no `index` provided."):
        sdt_to_DataFrame(TEMPORAL_DATA["sdt"])

    with pytest.raises(IndexError, match="The Dataset contains 1 measurements, but the "
                                         "indices supplied are 2."):
        sdt_to_DataFrame(TEMPORAL_DATA["sdt"], index=[1, 2])

    def bad_mapper(array: np.ndarray = None):
        return ((0,), (0,))

    with pytest.raises(ValueError,
                       match=r"The provided mapper wasn't sufficient, since the "
                             r"shape of the data is \(2, 524288\) and one value of the original "
                             r"shape \(64, 64, 256\) needs to be preserved."):
        sdt_to_DataFrame(FLIM_DATA["sdt"], mapper_function=bad_mapper)

    with pytest.raises(DimensionalityError,
                       match=r"The data you try to read are of shape \(64, 64, 256\), "
                             r"those data need to be flattened, which is done by "
                             r"utilizing a mapper function. The mapper function should "
                             r"provide the indices for the flattened data."):
        sdt_to_DataFrame(FLIM_DATA["sdt"])

    # this is just supposed to test the warning, which is why there is no need to bother
    # with the raised exception, due to the falsey values
    with pytest.raises(IndexError):
        with pytest.warns(UserWarning,
                          match=f"The file '{TEMPORAL_DATA['sdt']}' contains 2 Datasets.\n "
                                f"By default only the first Dataset will be read. "
                                f"If you only need the first Dataset and want get rid of "
                                f"this warning you can set dataset_index=0."):
            with monkeypatch.context() as m:

                class mocked_SdtFile():
                    def __init__(self, file_name):
                        self.times = [0, 0]
                        self.data = np.array([0, 0])

                        m.setattr(sdt_reader.SdtFile,
                                  "__init__",
                                  mocked_SdtFile.__init__)
                sdt_to_DataFrame(TEMPORAL_DATA["sdt"], index=[0, 1])


def test_sdt_to_df__flim():
    test_df, orig_shape = sdt_to_DataFrame(FLIM_DATA["sdt"], mapper_function=get_pixel_map)
    legacy_data, legacy_orig_shape = FLIM_legacy_to_DataFrame(FLIM_DATA["csv"], traces_only=False)
    linearized_intensity_map = legacy_data["intensity_map"].values.reshape(64*64)
    test_df_sum = np.sum(test_df.values, axis=1)

    assert orig_shape == legacy_orig_shape
    assert np.all(test_df_sum == linearized_intensity_map)
    assert np.allclose(test_df.columns, legacy_data["time_traces"].columns)

    selected_pixel_indices = legacy_data["time_traces"].index
    assert np.all(test_df.values[selected_pixel_indices] == legacy_data["time_traces"].values)


@pytest.mark.parametrize("spectral_unit", [
    'um', 'nm'
])
@pytest.mark.parametrize("time_unit", [
    's', 'ps'
])
@pytest.mark.parametrize("swap_axis, result_dict", [
    (False, {"data": [[1, 2], [3, 4]],
             "time_axis": [10, 20],
             "wl": [100, 200]}),
    (True, {"data": [[1, 3], [2, 4]],
            "time_axis": [100, 200],
            "wl": [10, 20]})
])
def test_df_to_SpectralTemporalDataset(swap_axis, result_dict, time_unit, spectral_unit):
    result = SpectralTemporalDataset(time_unit=time_unit)
    result.time_axis = np.array(result_dict["time_axis"])
    result.spectral_axis = np.array(result_dict["wl"])
    result.data = np.array(result_dict["data"])
    test_df = pd.DataFrame([[1, 2], [3, 4]], index=[100, 200], columns=[10, 20])
    test_dataset = DataFrame_to_SpectralTemporalDataset(test_df,
                                                        time_unit=time_unit,
                                                        spectral_unit=spectral_unit,
                                                        swap_axis=swap_axis)
    assert np.all(test_dataset.data == result.data)
    assert np.all(test_dataset.time_axis == result.time_axis)
    assert np.all(test_dataset.spectral_axis == result.spectral_axis)
    assert test_dataset.time_unit == time_unit
    assert test_dataset.spectral_unit == spectral_unit


@pytest.mark.parametrize("swap_axis", [
    False, True
])
def test_df_to_SpectralTemporalDataset__exceptions(swap_axis):
    with pytest.raises(ValueError,
                       match=f"The columns of the DataFrame needs to be convertible "
                             f"to numeric values."):
        test_df = pd.DataFrame([[0, 0], [0, 0]], columns=["foo", "bar"], index=[1, 2])
        DataFrame_to_SpectralTemporalDataset(test_df, swap_axis=swap_axis)

    with pytest.raises(ValueError,
                       match=f"The index of the DataFrame needs to be convertible "
                             f"to numeric values."):
        test_df = pd.DataFrame([[0, 0], [0, 0]], index=["foo", "bar"], columns=[1, 2])
        DataFrame_to_SpectralTemporalDataset(test_df, swap_axis=swap_axis)


@pytest.mark.parametrize("is_legacy", [
    True, False
])
@pytest.mark.parametrize("swap_axis", [
    True, False
])
@pytest.mark.parametrize("time_unit", [
    's', 'ps'
])
def test_df_to_FLIMDataset(is_legacy, swap_axis, time_unit):
    orig_shape = (2, 2, 3)
    result_array = np.arange(np.prod(orig_shape))
    result_array = result_array.reshape(orig_shape)
    result_intensity_map = pd.DataFrame(result_array.sum(axis=2))
    index = get_pixel_map(result_array)
    columns = np.arange(3)*0.1
    reshaped_result_array = result_array[get_pixel_map(result_array, transposed=True)]
    result_df = pd.DataFrame(reshaped_result_array, index=index, columns=columns)
    if swap_axis:
        test_df = result_df.T
    else:
        test_df = result_df
    if is_legacy:
        test_df = {"time_traces": test_df, "intensity_map": result_intensity_map}
    else:
        test_df = test_df

    test_dataset = DataFrame_to_FLIMDataset(test_df,
                                            mapper_function=get_pixel_map,
                                            orig_shape=orig_shape,
                                            time_unit=time_unit,
                                            swap_axis=swap_axis)

    assert isinstance(test_dataset, FLIMDataset)
    assert np.all(test_dataset.intensity_map == result_intensity_map)
    assert np.allclose(test_dataset.time_axis, np.array(result_df.columns))
    assert test_dataset.orig_shape == orig_shape
    assert test_dataset.data.shape == result_df.values.shape
    assert test_dataset.time_unit == time_unit


@pytest.mark.parametrize("swap_axis, index, columns, error_str", [
    (True, ["foo", "bar"], [1, 2], "index"),
    (False, [1, 2], ["foo", "bar"], "columns")
])
def test_df_to_FLIMDataset__exceptions(swap_axis, index, columns, error_str):
    with pytest.raises(ValueError,
                       match=f"The {error_str} of the DataFrame needs to be convertible "
                             f"to numeric values."):
        test_df = pd.DataFrame([[0, 0], [0, 0]], index=index, columns=columns)
        DataFrame_to_FLIMDataset(test_df, mapper_function=get_pixel_map,
                                 orig_shape=(2, 2), swap_axis=swap_axis)


@pytest.mark.parametrize("spectral_unit", [
    'um', 'nm'
])
@pytest.mark.parametrize("time_unit", [
    's', 'ps'
])
@pytest.mark.parametrize("return_dataframe", [
    True, False
])
@pytest.mark.parametrize("type_of_data, test_file_path, result_file_path, index", [
    ("st", TEMPORAL_DATA["sdt"], TEMPORAL_DATA["csv"], [1]),
    ("flim", FLIM_DATA["sdt"], FLIM_DATA["csv"], None)
])
def test_read_sdt(type_of_data, test_file_path, result_file_path, index, return_dataframe,
                  time_unit, spectral_unit):

    test_dataset = read_sdt(file_path=test_file_path, index=index,
                            type_of_data=type_of_data, time_unit=time_unit,
                            return_dataframe=return_dataframe, spectral_unit=spectral_unit)
    if type_of_data == "flim":
        result_dict, orig_shape = FLIM_legacy_to_DataFrame(FLIM_DATA["csv"],
                                                           traces_only=False, zero_pad=True)
        result_intensity_map = result_dict["intensity_map"]
        result_traces = result_dict["time_traces"]
        if not return_dataframe:
            assert isinstance(test_dataset, FLIMDataset)
            assert np.all(test_dataset.intensity_map == result_intensity_map)
            assert test_dataset.orig_shape == orig_shape
    else:
        result_df = pd.read_csv(result_file_path, skiprows=1, sep=r"\s+",
                                dtype={"Delay": np.float, "Data": np.uint16})
        result_df.Delay = result_df.Delay * 1e-9
        result_traces = pd.DataFrame([result_df.Data.values], columns=result_df.Delay)
        if not return_dataframe:
            assert isinstance(test_dataset, SpectralTemporalDataset)
            assert np.all(test_dataset.data[0] == result_df.Data.values)
            assert test_dataset.spectral_unit == spectral_unit

    if not return_dataframe:
        assert test_dataset.data.shape == result_traces.values.shape
        assert np.allclose(test_dataset.time_axis, np.array(result_traces.columns))
        assert test_dataset.time_unit == time_unit
    else:
        assert isinstance(test_dataset, pd.DataFrame)
        assert test_dataset.values.shape == result_traces.values.shape
        assert np.allclose(test_dataset.columns, np.array(result_traces.columns))


def test_read_sdt__exceptions():
    with pytest.raises(ValueError,
                       match=r"The entered value of `type_of_data` was 'not_supported', "
                             r"this value isn't supported. The supported values are "
                             r"\['st', 'flim'\]."):
        read_sdt("test_df", type_of_data='not_supported')
