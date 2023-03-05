import pytest
import xarray as xr

from glotaran.io.preprocessor import PreProcessingPipeline


def test_correct_baseline_value():
    pl = PreProcessingPipeline().correct_baseline_value(1)
    data = xr.DataArray([[1]])
    result = pl.apply(data)
    assert result == data - 1


@pytest.mark.parametrize("indexer", (slice(0, 2), [0, 1]))
def test_correct_baseline_average(indexer: slice | list[int]):
    pl = PreProcessingPipeline().correct_baseline_average(select={"dim_0": 0, "dim_1": indexer})
    data = xr.DataArray([[1.1, 0.9]])
    result = pl.apply(data)
    assert (result == data - 1).all()


def test_correct_baseline_average_exclude():
    pl = PreProcessingPipeline().correct_baseline_average(
        select={"dim_0": 0}, exclude={"dim_1": 1}
    )
    data = xr.DataArray([[1.1, 0.9]])
    result = pl.apply(data)
    print(result)
    assert (result == data - 1.1).all()


def test_to_from_dict():
    pl = (
        PreProcessingPipeline()
        .correct_baseline_value(1)
        .correct_baseline_average({"dim_1": slice(0, 2)})
    )
    pl_dict = pl.dict()
    assert pl_dict == {
        "actions": [
            {"action": "baseline-value", "value": 1.0},
            {"action": "baseline-average", "select": {"dim_1": slice(0, 2)}, "exclude": None},
        ]
    }
    assert PreProcessingPipeline.parse_obj(pl_dict) == pl
