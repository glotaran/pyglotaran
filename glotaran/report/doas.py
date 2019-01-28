
import typing
import functools
import holoviews as hv

from glotaran.analysis.fitresult import FitResult

from .kinetic import dataset as kdataset
from .saveable import saveable


@saveable
def dampened_oscillation_associated_spectra(result: FitResult, dataset: str) -> hv.Curve:
    curve = hv.Curve(result.data[dataset].dampened_oscillation_associated_spectra)\
        .groupby('oscillation').overlay()
    return curve.opts(title=f'DOAS {dataset}', legend_position='bottom_right')


@saveable
def dampened_oscillation_phase(result: FitResult, dataset: str) -> hv.Curve:
    curve = hv.Curve(result.data[dataset].dampened_oscillation_phase)\
        .groupby('oscillation').overlay()
    return curve.opts(title=f'Phase {dataset}', legend_position='bottom_right')


@saveable
def dataset(result: FitResult, dataset: str):
    layout = kdataset(result, dataset)
    layout += dampened_oscillation_associated_spectra(result, dataset)
    layout += dampened_oscillation_phase(result, dataset)
    return layout.cols(5)
