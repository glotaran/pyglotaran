import holoviews as hv
import xarray as xr

from glotaran.analysis.fitresult import FitResult
from glotaran.io import prepare_dataset

from .saveable import saveable


@saveable
def dataset(dataset: xr.Dataset, nr_svals: int, log_scale_time: bool = False):
    dataset = prepare_dataset(dataset)
    return _svd(dataset, 'data', nr_svals, log_scale_time)


@saveable
def residual(result: FitResult, dataset: str, nr_svals: int, log_scale_time: bool = False):
    dataset = result.data[dataset]
    return _svd(dataset, 'residual', nr_svals, log_scale_time)


def _svd(dataset: xr.Dataset, value: str, nr_svals: int, log_scale_time):

    nr_svals = min(nr_svals, dataset.coords['singular_value_index'].size)
    data = dataset[f'{value}_left_singular_vectors']
    lsv = hv.Curve(data.sel(left_singular_value_index=0))
    for i in range(1, nr_svals):
        lsv *= hv.Curve(data.sel(left_singular_value_index=i))

    lsv = lsv.options(hv.opts.Curve(logx=log_scale_time, framewise=True))

    data = dataset[f'{value}_right_singular_vectors']
    rsv = hv.Curve(data.sel(right_singular_value_index=0))
    for i in range(1, nr_svals):
        rsv *= hv.Curve(data.sel(right_singular_value_index=i))

    rsv = rsv.options(hv.opts.Curve(logx=log_scale_time, framewise=True))

    svals = hv.Curve(dataset[f'{value}_singular_values']).options(logy=True)

    return lsv + svals + rsv
