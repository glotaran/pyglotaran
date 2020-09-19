from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.parameter import ParameterGroup


def plot_overview(res, title=None):
    """ very simple plot helper function derived from pyglotaran_examples """
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16)
    sas = res.species_associated_spectra
    traces = res.species_concentration
    if "spectral" in traces.coords:
        traces.sel(spectral=res.spectral.values[0], method="nearest").plot.line(
            x="time", ax=ax[0, 0]
        )
    else:
        traces.plot.line(x="time", ax=ax[0, 0])
    sas.plot.line(x="spectral", ax=ax[0, 1])
    rLSV = res.residual_left_singular_vectors
    rLSV.isel(left_singular_value_index=range(0, min(2, len(rLSV)))).plot.line(
        x="time", ax=ax[1, 0]
    )
    ax[1, 0].set_title("res. LSV")
    rRSV = res.residual_right_singular_vectors
    rRSV.isel(right_singular_value_index=range(0, min(2, len(rRSV)))).plot.line(
        x="spectral", ax=ax[1, 1]
    )
    ax[1, 1].set_title("res. RSV")
    ax[0, 2].set_title("data")
    res.data.plot(x="time", ax=ax[0, 2])
    plt.show(block=False)


def test_spectral_penalties():

    mspec_base = {
        "initial_concentration": {
            "j1": {"compartments": ["s1", "s2"], "parameters": ["i.1", "i.2"]},
        },
        "megacomplex": {
            "mc1": {"k_matrix": ["k1"]},
        },
        "k_matrix": {
            "k1": {
                "matrix": {
                    ("s1", "s1"): "kinetic.1",
                    ("s2", "s2"): "kinetic.2",
                }
            }
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": "j1",
                "megacomplex": ["mc1"],
            },
        },
        # "spectral_constraints": [
        #     {
        #         "type": "zero",
        #         "compartment": "s1",
        #         "interval": [(7, 9)],
        #     },
        #     {
        #         "type": "zero",
        #         "compartment": "s2",
        #         "interval": [(5, 6)],
        #     },
        # ]
    }

    mspec_equ_area = {
        "equal_area_penalties": [
            {
                "compartment": "s2",
                "target": "s1",
                "parameter": "pen.1",
                "interval": [(0, 90)],
                "weight": 100,
            },
        ],
    }
    mspec_np = deepcopy(mspec_base)
    mspec_wp = dict(deepcopy(mspec_base), **mspec_equ_area)

    model_without_penalty = KineticSpectrumModel.from_dict(mspec_np)
    model_with_penalty = KineticSpectrumModel.from_dict(mspec_wp)

    pspec_base = {
        "kinetic": [0.02, 0.1, 0.5],
        "i": [1.0, 1.0, {"vary": False}],
        "pen": [1, {"vary": False}],
    }
    pspec_fit = {"i": [[1, {"vary": True}], 3]}

    pspec_np = dict(pspec_base, **pspec_fit)
    del pspec_np["pen"]
    pspec_wp = dict(pspec_base, **pspec_fit)

    param_sim = ParameterGroup.from_dict(deepcopy(pspec_base))
    param_np = ParameterGroup.from_dict(deepcopy(pspec_np))
    param_wp = ParameterGroup.from_dict(deepcopy(pspec_wp))

    time = np.asarray(np.arange(0, 99, 0.5))
    clp = xr.DataArray(
        [[3.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        coords=(("spectral", [5.0, 6.0, 7.0, 8.0]), ("clp_label", ["s1", "s2"])),
    )

    data = model_without_penalty.simulate(
        "dataset1",
        param_sim,
        clp=clp,
        axes={"time": time, "spectral": clp.spectral.values},
        noise=True,
        noise_std_dev=1e-8,
        noise_seed=1,
    )

    result_np = model_without_penalty.optimize(param_np, {"dataset1": data}, nnls=True)
    plot_overview(result_np.data["dataset1"], "without penalties")

    result_wp = model_with_penalty.optimize(param_wp, {"dataset1": data}, nnls=True)
    plot_overview(result_wp.data["dataset1"], "with penalties")

    result_data = result_wp.data["dataset1"]

    area1 = np.sum(result_data.species_associated_spectra.sel(species="s1"))
    area2 = np.sum(result_data.species_associated_spectra.sel(species="s2"))
    print(f"area1: {area1}\narea2: {area2}\n")
    plt.show()


if __name__ == "__main__":
    test_spectral_penalties()
