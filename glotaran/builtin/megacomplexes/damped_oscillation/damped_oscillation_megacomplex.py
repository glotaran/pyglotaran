from __future__ import annotations

from typing import List

import numba as nb
import numpy as np
import xarray as xr
from scipy.special import erf

from glotaran.builtin.megacomplexes.decay.irf import Irf
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import megacomplex
from glotaran.model.item import model_item_validator
from glotaran.parameter import Parameter


@megacomplex(
    dimension="time",
    dataset_model_items={
        "irf": {"type": Irf, "allow_none": True},
    },
    properties={
        "labels": List[str],
        "frequencies": List[Parameter],
        "rates": List[Parameter],
    },
    register_as="damped-oscillation",
)
class DampedOscillationMegacomplex(Megacomplex):
    @model_item_validator(False)
    def ensure_oscillation_paramater(self, model: Model) -> list[str]:

        problems = []

        if len(self.labels) != len(self.frequencies) or len(self.labels) != len(self.rates):
            problems.append(
                f"Size of labels ({len(self.labels)}), frequencies ({len(self.frequencies)}) "
                f"and rates ({len(self.rates)}) does not match for damped oscillation "
                f"megacomplex '{self.label}'."
            )

        return problems

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        indices: dict[str, int],
        **kwargs,
    ):

        clp_label = [f"{label}_cos" for label in self.labels] + [
            f"{label}_sin" for label in self.labels
        ]

        model_axis = dataset_model.get_model_axis()
        delta = np.abs(model_axis[1:] - model_axis[:-1])
        delta_min = delta[np.argmin(delta)]
        frequency_max = 1 / (2 * 0.03 * delta_min)
        frequencies = np.array(self.frequencies) * 0.03 * 2 * np.pi
        frequencies[frequencies >= frequency_max] = np.mod(
            frequencies[frequencies >= frequency_max], frequency_max
        )
        rates = np.array(self.rates)

        matrix = np.ones((model_axis.size, len(clp_label)), dtype=np.float64)

        if dataset_model.irf is None:
            calculate_damped_oscillation_matrix_no_irf(matrix, frequencies, rates, model_axis)
        elif isinstance(dataset_model.irf, IrfMultiGaussian):
            global_dimension = dataset_model.get_global_dimension()
            global_axis = dataset_model.get_global_axis()
            global_index = indices.get(global_dimension)
            centers, widths, scales, shift, _, _ = dataset_model.irf.parameter(
                global_index, global_axis
            )
            for center, width, scale in zip(centers, widths, scales):
                matrix += calculate_damped_oscillation_matrix_gaussian_irf(
                    frequencies,
                    rates,
                    model_axis,
                    center,
                    width,
                    shift,
                    scale,
                )
            matrix /= np.sum(scales)

        return clp_label, matrix

    def index_dependent(self, dataset_model: DatasetModel) -> bool:
        return (
            isinstance(dataset_model.irf, IrfMultiGaussian)
            and dataset_model.irf.is_index_dependent()
        )

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        if is_full_model:
            return

        megacomplexes = (
            dataset_model.global_megacomplex if is_full_model else dataset_model.megacomplex
        )
        unique = len([m for m in megacomplexes if isinstance(m, DampedOscillationMegacomplex)]) < 2

        prefix = "damped_oscillation" if unique else f"{self.label}_damped_oscillation"

        dataset.coords[f"{prefix}"] = self.labels
        dataset.coords[f"{prefix}_frequency"] = (prefix, self.frequencies)
        dataset.coords[f"{prefix}_rate"] = (prefix, self.rates)

        dim1 = dataset_model.get_global_axis().size
        dim2 = len(self.labels)
        doas = np.zeros((dim1, dim2), dtype=np.float64)
        phase = np.zeros((dim1, dim2), dtype=np.float64)
        for i, label in enumerate(self.labels):
            sin = dataset.clp.sel(clp_label=f"{label}_sin")
            cos = dataset.clp.sel(clp_label=f"{label}_cos")
            doas[:, i] = np.sqrt(sin * sin + cos * cos)
            phase[:, i] = np.unwrap(np.arctan2(sin, cos))

        dataset[f"{prefix}_associated_spectra"] = (
            (dataset_model.get_global_dimension(), prefix),
            doas,
        )

        dataset[f"{prefix}_phase"] = (
            (dataset_model.get_global_dimension(), prefix),
            phase,
        )

        if not is_full_model:
            if self.index_dependent(dataset_model):
                dataset[f"{prefix}_sin"] = (
                    (
                        dataset_model.get_global_dimension(),
                        dataset_model.get_model_dimension(),
                        prefix,
                    ),
                    dataset.matrix.sel(clp_label=[f"{label}_sin" for label in self.labels]).values,
                )

                dataset[f"{prefix}_cos"] = (
                    (
                        dataset_model.get_global_dimension(),
                        dataset_model.get_model_dimension(),
                        prefix,
                    ),
                    dataset.matrix.sel(clp_label=[f"{label}_cos" for label in self.labels]).values,
                )
            else:
                dataset[f"{prefix}_sin"] = (
                    (dataset_model.get_model_dimension(), prefix),
                    dataset.matrix.sel(clp_label=[f"{label}_sin" for label in self.labels]).values,
                )

                dataset[f"{prefix}_cos"] = (
                    (dataset_model.get_model_dimension(), prefix),
                    dataset.matrix.sel(clp_label=[f"{label}_cos" for label in self.labels]).values,
                )


@nb.jit(nopython=True, parallel=True)
def calculate_damped_oscillation_matrix_no_irf(matrix, frequencies, rates, axis):

    idx = 0
    for frequency, rate in zip(frequencies, rates):
        osc = np.exp(-rate * axis - 1j * frequency * axis)
        matrix[:, idx] = osc.real
        matrix[:, idx + 1] = osc.imag
        idx += 2


def calculate_damped_oscillation_matrix_gaussian_irf(
    frequencies: np.ndarray,
    rates: np.ndarray,
    model_axis: np.ndarray,
    center: float,
    width: float,
    shift: float,
    scale: float,
    show_debug_plot: bool = False,  # TODO: remove
):
    shifted_axis = model_axis - center - shift
    d = width ** 2
    k = rates + 1j * frequencies
    dk = k * d
    sqwidth = np.sqrt(2) * width
    a = (-1 * shifted_axis[:, None] + 0.5 * dk) * k
    a = np.minimum(a, 709)
    a = np.exp(a)
    b = 1 + erf((shifted_axis[:, None] - dk) / sqwidth)
    osc = a * b * scale
    # Temporary debug plotting code
    if show_debug_plot:
        _debug_plot_osc_a_b_decomposition(shifted_axis, a, b, osc)
        _debug_plot_osc_real_imag(shifted_axis, osc)
        show_debug_plot = False

    return np.concatenate((osc.real, osc.imag), axis=1)


def _debug_plot_osc_real_imag(shifted_axis, osc, time_slice=slice(24, 124)):
    # TODO: remove debugging code
    # For debugging only
    import matplotlib.pyplot as plt
    from pyglotaran_extras.plotting.style import PlotStyle

    plot_style = PlotStyle()
    plt.rc("axes", prop_cycle=plot_style.cycler)

    _, axes = plt.subplots(5, 2, figsize=(25, 25))

    time_axis = shifted_axis[time_slice]
    axes[0, 0].plot(time_axis, osc.real[time_slice, 14:])
    axes[0, 1].plot(time_axis, osc.imag[time_slice, 14:])
    axes[1, 0].plot(time_axis, osc.real[time_slice, 9:14])
    axes[1, 1].plot(time_axis, osc.imag[time_slice, 9:14])
    axes[2, 0].plot(time_axis, osc.real[time_slice, 6:9])
    axes[2, 1].plot(time_axis, osc.imag[time_slice, 6:9])
    axes[3, 0].plot(time_axis, osc.real[time_slice, 3:6])
    axes[3, 1].plot(time_axis, osc.imag[time_slice, 3:6])
    axes[4, 0].plot(time_axis, osc.real[time_slice, 0:3])
    axes[4, 1].plot(time_axis, osc.imag[time_slice, 0:3])
    # plt.rc("axes", prop_cycle=plot_style.cycler)
    plt.show()


def _debug_plot_osc_a_b_decomposition(shifted_axis, a, b, osc, time_slice=slice(24, 124)):
    # TODO: remove debugging code
    # For debugging only
    import matplotlib.pyplot as plt
    from pyglotaran_extras.plotting.style import PlotStyle

    plot_style = PlotStyle()
    _, axes = plt.subplots(4, 3, figsize=(25, 25))
    axes[0, 0].plot()
    plt.rc("axes", prop_cycle=plot_style.cycler)
    time_axis = shifted_axis[time_slice]
    axes[0, 0].plot(time_axis, a[time_slice, :14])
    axes[1, 0].plot(time_axis, b[time_slice, :14])
    axes[2, 0].plot(time_axis, osc.real[time_slice, :14])
    axes[3, 0].plot(time_axis, osc.imag[time_slice, :14])
    axes[0, 1].plot(time_axis, a[time_slice, 14:])
    axes[1, 1].plot(time_axis, b[time_slice, 14:])
    axes[2, 1].plot(time_axis, osc.real[time_slice, 14:])
    axes[3, 1].plot(time_axis, osc.imag[time_slice, 14:])
    axes[0, 2].plot(time_axis, a[time_slice, :])
    axes[1, 2].plot(time_axis, b[time_slice, :])
    axes[2, 2].plot(time_axis, osc.real[time_slice, :])
    axes[3, 2].plot(time_axis, osc.imag[time_slice, :])
    plt.rc("axes", prop_cycle=plot_style.cycler)
    cols = [f"{col} rates" for col in ["positive", "negative", "all"]]
    rows = [f"{row}" for row in ["a", "b", "osc.\nreal", "osc.\nimag"]]
    pad = 5  # in points
    for ax, col in zip(axes[0], cols):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, 4 * pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
            bbox={"facecolor": "none", "edgecolor": "black", "boxstyle": "round,pad=1"},
        )

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
            bbox={"facecolor": "none", "edgecolor": "black", "boxstyle": "round,pad=1"},
        )
    plt.show()
