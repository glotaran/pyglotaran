import numpy as np
import scipy

from glotaran.analysis.result import Result
from glotaran.models.spectral_temporal.kinetic_fit_result import finalize_kinetic_result


def finalize_doas_result(model, result: Result):

    finalize_kinetic_result(model, result)

    for label in result.model.dataset:
        dataset = result.data[label]

        dataset_descriptor = result.model.dataset[label].fill(model, result.best_fit_parameter)

        # get_doas

        oscillations = []

        for cmplx in dataset_descriptor.megacomplex:
            for osc in cmplx.oscillation:
                if osc.label not in oscillations:
                    oscillations.append(osc.label)

        dim1 = dataset.coords[model.estimated_axis].size
        dim2 = len(oscillations)
        doas = np.zeros((dim1, dim2), dtype=np.float64)
        phase = np.zeros((dim1, dim2), dtype=np.float64)
        for i, osc in enumerate(oscillations):
            sin = dataset.clp.sel(clp_label=f'{osc}_sin')
            cos = dataset.clp.sel(clp_label=f'{osc}_cos')
            doas[:, i] = np.sqrt(sin*sin+cos*cos)
            phase[:, i] = np.unwrap(np.arctan2(cos, sin))

        dataset.coords['oscillation'] = oscillations

        dataset['dampened_oscillation_associated_spectra'] = (
            (model.estimated_axis, 'oscillation'), doas)

        dataset['dampened_oscillation_phase'] = (
            (model.estimated_axis, 'oscillation'), phase)

        dataset['dampened_oscillation_concentration_sin'] = (
            (model.estimated_axis, model.calculated_axis, 'oscillation'),
            dataset.concentration.sel(clp_label=[f'{osc}_sin' for osc in oscillations])
        )

        dataset['dampened_oscillation_concentration_cos'] = (
            (model.estimated_axis, model.calculated_axis, 'oscillation'),
            dataset.concentration.sel(clp_label=[f'{osc}_cos' for osc in oscillations])
        )

    time_diff = np.diff(dataset.time, n=1, axis=0)

    power = dataset.residual_left_singular_vectors.isel(left_singular_value_index=0).values[:-1]
    power = power[time_diff < time_diff.mean()]

    power = scipy.fftpack.fft(power, n=1024, axis=0)

    power = np.abs(power)/power.size

    dataset['residual_power_spectrum'] = (('frequency'), power)
