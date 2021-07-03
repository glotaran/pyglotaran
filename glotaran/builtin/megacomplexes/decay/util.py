from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.builtin.megacomplexes.decay.irf import IrfSpectralMultiGaussian
from glotaran.model import DatasetModel

if TYPE_CHECKING:
    from glotaran.builtin.megacomplexes.decay.decay_megacomplex import DecayMegacomplex


def decay_matrix_implementation(
    matrix: np.ndarray,
    rates: np.ndarray,
    global_index: int,
    global_axis: np.ndarray,
    model_axis: np.ndarray,
    dataset_model: DatasetModel,
):
    if isinstance(dataset_model.irf, IrfMultiGaussian):

        (
            centers,
            widths,
            irf_scales,
            shift,
            backsweep,
            backsweep_period,
        ) = dataset_model.irf.parameter(global_index, global_axis)

        for center, width, irf_scale in zip(centers, widths, irf_scales):
            calculate_decay_matrix_gaussian_irf(
                matrix,
                rates,
                model_axis,
                center - shift,
                width,
                irf_scale,
                backsweep,
                backsweep_period,
            )
        if dataset_model.irf.normalize:
            matrix /= np.sum(irf_scale)

    else:
        calculate_decay_matrix_no_irf(matrix, rates, model_axis)


@nb.jit(nopython=True, parallel=True)
def calculate_decay_matrix_no_irf(matrix, rates, times):
    for n_r in nb.prange(rates.size):
        r_n = rates[n_r]
        for n_t in range(times.size):
            t_n = times[n_t]
            matrix[n_t, n_r] += np.exp(r_n * t_n)


sqrt2 = np.sqrt(2)


@nb.jit(nopython=True, parallel=True)
def calculate_decay_matrix_gaussian_irf(
    matrix, rates, times, center, width, scale, backsweep, backsweep_period
):
    """Calculates a decay matrix with a gaussian irf."""
    for n_r in nb.prange(rates.size):
        r_n = -rates[n_r]
        backsweep_valid = abs(r_n) * backsweep_period > 0.001
        alpha = (r_n * width) / sqrt2
        for n_t in nb.prange(times.size):
            t_n = times[n_t]
            beta = (t_n - center) / (width * sqrt2)
            thresh = beta - alpha
            if thresh < -1:
                matrix[n_t, n_r] += scale * 0.5 * erfcx(-thresh) * np.exp(-beta * beta)
            else:
                matrix[n_t, n_r] += (
                    scale * 0.5 * (1 + erf(thresh)) * np.exp(alpha * (alpha - 2 * beta))
                )
            if backsweep and backsweep_valid:
                x1 = np.exp(-r_n * (t_n - center + backsweep_period))
                x2 = np.exp(-r_n * ((backsweep_period / 2) - (t_n - center)))
                x3 = np.exp(-r_n * backsweep_period)
                matrix[n_t, n_r] += scale * (x1 + x2) / (1 - x3)


import ctypes  # noqa: E402

# This is a work around to use scipy.special function with numba
from numba.extending import get_cython_function_address  # noqa: E402

_dble = ctypes.c_double

functype = ctypes.CFUNCTYPE(_dble, _dble)

erf_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erf")
erfcx_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erfcx")

erf = functype(erf_addr)
erfcx = functype(erfcx_addr)


def retrieve_species_associated_data(
    dataset_model: DatasetModel, data: xr.Dataset, global_dimension: str, name: str
):
    species = dataset_model.initial_concentration.compartments
    model_dimension = dataset_model.get_model_dimension()

    data.coords["species"] = species
    data[f"species_associated_{name}"] = (
        (
            global_dimension,
            "species",
        ),
        data.clp.sel(clp_label=species).data,
    )

    if len(data.matrix.shape) == 3:
        #  index dependent
        data["species_concentration"] = (
            (
                global_dimension,
                model_dimension,
                "species",
            ),
            data.matrix.sel(clp_label=species).values,
        )
    else:
        #  index independent
        data["species_concentration"] = (
            (
                model_dimension,
                "species",
            ),
            data.matrix.sel(clp_label=species).values,
        )


def retrieve_decay_associated_data(
    megacomplex: DecayMegacomplex,
    dataset_model: DatasetModel,
    data: xr.Dataset,
    global_dimension: str,
    name: str,
    multiple_complexes: bool,
):
    k_matrix = megacomplex.full_k_matrix()

    species = dataset_model.initial_concentration.compartments
    species = [c for c in species if c in k_matrix.involved_compartments()]

    matrix = k_matrix.full(species)
    matrix_reduced = k_matrix.reduced(species)
    a_matrix = k_matrix.a_matrix(dataset_model.initial_concentration)
    rates = k_matrix.rates(dataset_model.initial_concentration)
    lifetimes = 1 / rates

    das = data[f"species_associated_{name}"].sel(species=species).values @ a_matrix.T

    component_coords = {"rate": ("component", rates), "lifetime": ("component", lifetimes)}
    das_coords = component_coords.copy()
    das_coords[global_dimension] = data.coords[global_dimension]
    das_name = f"decay_associated_{name}"
    das = xr.DataArray(das, dims=(global_dimension, "component"), coords=das_coords)

    a_matrix_coords = component_coords.copy()
    a_matrix_coords["species"] = species
    a_matrix_name = "a_matrix"
    a_matrix = xr.DataArray(a_matrix, coords=a_matrix_coords, dims=("component", "species"))

    k_matrix_name = "k_matrix"
    k_matrix = xr.DataArray(matrix, coords=[("to_species", species), ("from_species", species)])

    k_matrix_reduced_name = "k_matrix_reduced"
    k_matrix_reduced = xr.DataArray(
        matrix_reduced, coords=[("to_species", species), ("from_species", species)]
    )

    if multiple_complexes:
        das_name = f"decay_associated_{name}_{megacomplex.label}"
        das = das.rename(component=f"component_{megacomplex.label}")
        a_matrix_name = f"a_matrix_{megacomplex.label}"
        a_matrix = a_matrix.rename(component=f"component_{megacomplex.label}")
        k_matrix_name = f"k_matrix_{megacomplex.label}"
        k_matrix_reduced_name = f"k_matrix_reduced_{megacomplex.label}"

    data[das_name] = das
    data[a_matrix_name] = a_matrix
    data[k_matrix_name] = k_matrix
    data[k_matrix_reduced_name] = k_matrix_reduced


def retrieve_irf(dataset_model: DatasetModel, data: xr.Dataset, global_dimension: str):

    irf = dataset_model.irf
    model_dimension = dataset_model.get_model_dimension()

    data["irf"] = (
        (model_dimension),
        irf.calculate(
            index=0,
            global_axis=data.coords[global_dimension].values,
            model_axis=data.coords[model_dimension].values,
        ).data,
    )
    data["irf_center"] = irf.center
    data["irf_width"] = irf.width
    if isinstance(irf, IrfSpectralMultiGaussian) and irf.dispersion_center:
        for i, dispersion in enumerate(irf.calculate_dispersion(data.coords["spectral"].values)):
            data[f"center_dispersion_{i+1}"] = (
                global_dimension,
                dispersion,
            )
