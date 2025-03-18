from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from glotaran.builtin.io.ascii.wavelength_time_explicit_file import ExplicitFile

DATA_DIR = Path(__file__).parent
TEST_FILE_ASCII = DATA_DIR.joinpath("data.ascii")

TEST_BLOCK_2x2_TOP_LEFT = np.array(
    [[9.04130489e-02, -3.17128114e-02], [-8.30020159e-02, -2.86547579e-02]]
)
TEST_BLOCK_2x2_BOTTOM_RIGHT = np.array([[0.384812891, 0.342516541], [0.350679725, 0.371651471]])


def test_read_explicit_file():
    data_file = ExplicitFile(TEST_FILE_ASCII)
    test_dataarray = data_file.read(prepare=False)
    assert isinstance(test_dataarray, xr.DataArray)
    assert test_dataarray.shape == (501, 51)
    assert np.array_equal(TEST_BLOCK_2x2_TOP_LEFT, test_dataarray[:2, :2].to_numpy().T)
    assert np.array_equal(TEST_BLOCK_2x2_BOTTOM_RIGHT, test_dataarray[-2:, -2:].to_numpy().T)
    test_dataset = data_file.read()
    assert isinstance(test_dataset, xr.Dataset)
    assert min(test_dataset.coords["spectral"].values) == 400
    assert max(test_dataset.coords["spectral"].values) == 600
    assert min(test_dataset.coords["time"].values) == 0.0
    assert max(test_dataset.coords["time"].values) == 5.0
    test_dataset.sel(spectral=[620, 630, 650], method="nearest")


def test_write_explicit_file(tmp_path: Path):
    data_file = ExplicitFile(TEST_FILE_ASCII)
    test_dataarray_read = data_file.read(prepare=False)
    test_data_file = tmp_path.joinpath("test.ascii")
    test_data_file_write = ExplicitFile(filepath=str(test_data_file), dataset=test_dataarray_read)
    test_data_file_write.write(comment="written \n in \n test.", overwrite=True)
    test_dataarray_reread = test_data_file_write.read(prepare=False)
    assert np.array_equal(test_dataarray_read.values, test_dataarray_reread.values)
