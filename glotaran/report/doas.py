
import holoviews as hv

from glotaran.analysis.result import Result

from .kinetic import dataset as kdataset
from .saveable import saveable


@saveable
def dampened_oscillation_associated_spectra(result: Result, dataset: str) -> hv.Curve:
    curve = hv.Curve(result.data[dataset].dampened_oscillation_associated_spectra)\
        .groupby('oscillation').overlay()
    return curve.opts(title=f'DOAS {dataset}')


@saveable
def dampened_oscillation_phase(result: Result, dataset: str) -> hv.Curve:
    curve = hv.Curve(result.data[dataset].dampened_oscillation_phase)\
        .groupby('oscillation').overlay()
    return curve.opts(title=f'Phase {dataset}')


@saveable
def dataset(result: Result, dataset: str):
    layout = kdataset(result, dataset)
    layout += dampened_oscillation_associated_spectra(result, dataset)
    layout += dampened_oscillation_phase(result, dataset)
    return layout.cols(5)
