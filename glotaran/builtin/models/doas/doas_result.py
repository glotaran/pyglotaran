import typing
import numpy as np
import xarray as xr
from scipy import fftpack

import glotaran
from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_result import \
    finalize_kinetic_spectrum_result


def finalize_doas_data(
    model: 'glotaran.models.doas.DoasModel',
    global_indices: typing.List[typing.List[object]],
    reduced_clp_labels: typing.Union[typing.Dict[str, typing.List[str]], np.ndarray],
    reduced_clps: typing.Union[typing.Dict[str, np.ndarray], np.ndarray],
    parameter: ParameterGroup, data: typing.Dict[str, xr.Dataset],
):

    finalize_kinetic_spectrum_result(
        model, global_indices, reduced_clp_labels, reduced_clps, parameter, data)

    for label in model.dataset:
        dataset = data[label]

        dataset_descriptor = model.dataset[label].fill(model, parameter)

        # get_doas

        oscillations = []

        for cmplx in dataset_descriptor.megacomplex:
            for osc in cmplx.oscillation:
                if osc.label not in oscillations:
                    oscillations.append(osc.label)

        dim1 = dataset.coords[model.global_dimension].size
        dim2 = len(oscillations)
        doas = np.zeros((dim1, dim2), dtype=np.float64)
        phase = np.zeros((dim1, dim2), dtype=np.float64)
        for i, osc in enumerate(oscillations):
            sin = dataset.clp.sel(clp_label=f'{osc}_sin')
            cos = dataset.clp.sel(clp_label=f'{osc}_cos')
            doas[:, i] = np.sqrt(sin*sin+cos*cos)
            phase[:, i] = np.unwrap(np.arctan2(sin, cos))

        dataset.coords['oscillation'] = oscillations

        dataset['dampened_oscillation_associated_spectra'] = (
            (model.global_dimension, 'oscillation'), doas)

        dataset['dampened_oscillation_phase'] = (
            (model.global_dimension, 'oscillation'), phase)

        dataset['dampened_oscillation_sin'] = \
            dataset.matrix.sel(clp_label=[f'{osc}_sin' for osc in oscillations])\
            .rename(clp_label='oscillation')

        dataset['dampened_oscillation_cos'] = \
            dataset.matrix.sel(clp_label=[f'{osc}_cos' for osc in oscillations])\
            .rename(clp_label='oscillation')

    time_diff = np.diff(dataset.time, n=1, axis=0)

    power = dataset.residual_left_singular_vectors.isel(left_singular_value_index=0).values[:-1]
    power = power[time_diff < time_diff.mean()]

    power = fftpack.fft(power, n=1024, axis=0)

    power = np.abs(power)/power.size

    dataset['residual_power_spectrum'] = (('frequency'), power)
