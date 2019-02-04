import typing
import functools
import holoviews as hv

from glotaran.analysis.result import Result

from .saveable import saveable


@saveable
def decay_associated_spectra(result: Result, dataset: str) -> hv.Curve:
    megacomplexes = result.data[dataset].coords['megacomplex'].values
    curve = \
        hv.Curve(result.data[dataset].decay_associated_spectra.sel(megacomplex=megacomplexes[0]))\
        .groupby('compartment').overlay()
    for m in megacomplexes[1:]:
        curve += hv.Curve(result.data[dataset].decay_associated_spectra.sel(megacomplex=m))\
            .groupby('compartment').overlay()
    return curve.opts(title=f'DAS {dataset}')


@saveable
def irf_dispersion(result: Result, dataset: str) -> hv.Curve:
    megacomplexes = result.data[dataset].coords['megacomplex'].values
    curve = \
        hv.Curve(result.data[dataset].decay_associated_spectra.sel(megacomplex=megacomplexes[0]))\
        .groupby('compartment').overlay()
    for m in megacomplexes[1:]:
        curve += hv.Curve(result.data[dataset].decay_associated_spectra.sel(megacomplex=m))\
            .groupby('compartment').overlay()
    return curve.opts(label=f'IrfDispersion {dataset}')


@saveable
def species_concentration(result: Result, dataset: str, index: float) -> hv.Curve:
    return hv.Curve(
        result.data[dataset].species_concentration.sel(spectral=index, method='nearest'))\
        .groupby('species').overlay().opts(title=f'Concentration {dataset}')


@saveable
def species_associated_spectra(result: Result, dataset: str) -> hv.Curve:
    return hv.Curve(result.data[dataset].species_associated_spectra)\
        .groupby('species').overlay().opts(title=f'SAS {dataset}')


@saveable
def dataset(result: Result, dataset: str):
    layout = species_concentration(result, dataset, 0)
    layout += species_associated_spectra(result, dataset)
    layout += decay_associated_spectra(result, dataset)
    return layout.cols(3)
