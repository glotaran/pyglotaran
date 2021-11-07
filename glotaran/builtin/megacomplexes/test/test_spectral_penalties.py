from __future__ import annotations

import importlib
from collections import namedtuple
from copy import deepcopy

import numpy as np

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.builtin.megacomplexes.spectral import SpectralMegacomplex
from glotaran.io import prepare_time_trace_dataset
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme

ParameterSpec = namedtuple("ParameterSpec", "base equal_area shapes")
NoiseSpec = namedtuple("NoiseSpec", "active seed std_dev")
SimulationSpec = namedtuple("SimulationSpec", "max_nfev noise")
DatasetSpec = namedtuple("DatasetSpec", "times wavelengths irf shapes")
IrfSpec = namedtuple("IrfSpec", "location width")
ShapeSpec = namedtuple("ShapeSpec", "amplitude location width")
ModelSpec = namedtuple("ModelSpec", "base shape spectral_megacomplex equ_area")
OptimizationSpec = namedtuple("OptimizationSpec", "nnls max_nfev")


class SpectralDecayModel(Model):
    @classmethod
    def from_dict(
        cls,
        model_dict,
        *,
        megacomplex_types: dict[str, type[Megacomplex]] | None = None,
        default_megacomplex_type: str | None = None,
    ):
        defaults: dict[str, type[Megacomplex]] = {
            "decay": DecayMegacomplex,
            "spectral": SpectralMegacomplex,
        }
        if megacomplex_types is not None:
            defaults.update(megacomplex_types)
        return super().from_dict(
            model_dict,
            megacomplex_types=defaults,
            default_megacomplex_type=default_megacomplex_type,
        )


def plot_overview(res, title=None):
    """very simple plot helper function derived from pyglotaran_extras"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16)
    sas = res.species_associated_spectra
    traces = res.species_concentration
    if "spectral" in traces.coords:
        traces.sel(spectral=res.spectral[0], method="nearest").plot.line(x="time", ax=ax[0, 0])
    else:
        traces.plot.line(x="time", ax=ax[0, 0])
    sas.plot.line(x="spectral", ax=ax[0, 1])
    rLSV = res.residual_left_singular_vectors
    rLSV.isel(left_singular_value_index=range(min(2, len(rLSV)))).plot.line(x="time", ax=ax[1, 0])

    ax[1, 0].set_title("res. LSV")
    rRSV = res.residual_right_singular_vectors
    rRSV.isel(right_singular_value_index=range(min(2, len(rRSV)))).plot.line(
        x="spectral", ax=ax[1, 1]
    )

    ax[1, 1].set_title("res. RSV")
    plt.show(block=False)


def test_equal_area_penalties(debug=False):
    # %%

    optim_spec = OptimizationSpec(nnls=True, max_nfev=999)
    noise_spec = NoiseSpec(active=True, seed=1, std_dev=1e-8)

    wavelengths = np.arange(650, 670, 2)
    time_p1 = np.linspace(-1, 2, 50, endpoint=False)
    time_p2 = np.linspace(2, 10, 30, endpoint=False)
    time_p3 = np.geomspace(10, 50, num=20)
    times = np.concatenate([time_p1, time_p2, time_p3])

    irf_loc = float(times[20])
    irf_width = float((times[1] - times[0]) * 10)
    irf = IrfSpec(irf_loc, irf_width)

    amplitude = 1
    location1 = float(wavelengths[2])  # 2
    location2 = float(wavelengths[-3])  # -3
    width1 = float((wavelengths[1] - wavelengths[0]) * 5)
    width2 = float((wavelengths[1] - wavelengths[0]) * 3)
    shape1 = ShapeSpec(amplitude, location1, width1)
    shape2 = ShapeSpec(amplitude, location2, width2)
    dataset_spec = DatasetSpec(times, wavelengths, irf, [shape1, shape2])

    wavelengths = dataset_spec.wavelengths
    equ_interval = [(min(wavelengths), max(wavelengths))]
    weight = 0.01
    # %% The base model specification (mspec)
    base = {
        "initial_concentration": {
            "j1": {
                "compartments": ["s1", "s2"],
                "parameters": ["i.1", "i.2"],
            },
        },
        "megacomplex": {
            "mc1": {"type": "decay", "k_matrix": ["k1"]},
        },
        "k_matrix": {
            "k1": {
                "matrix": {
                    ("s1", "s1"): "kinetic.1",
                    ("s2", "s2"): "kinetic.2",
                }
            }
        },
        "irf": {
            "irf1": {"type": "gaussian", "center": "irf.center", "width": "irf.width"},
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": "j1",
                "megacomplex": ["mc1"],
                "irf": "irf1",
            },
        },
    }

    shape = {
        "shape": {
            "sh1": {
                "type": "gaussian",
                "amplitude": "shapes.amps.1",
                "location": "shapes.locs.1",
                "width": "shapes.width.1",
            },
            "sh2": {
                "type": "gaussian",
                "amplitude": "shapes.amps.2",
                "location": "shapes.locs.2",
                "width": "shapes.width.2",
            },
        }
    }

    spectral_megacomplex = {
        "type": "spectral",
        "shape": {
            "s1": "sh1",
            "s2": "sh2",
        },
    }

    equ_area = {
        "clp_area_penalties": [
            {
                "source": "s1",
                "target": "s2",
                "parameter": "rela.1",
                "source_intervals": equ_interval,
                "target_intervals": equ_interval,
                "weight": weight,
            },
        ],
    }
    mspec = ModelSpec(base, shape, spectral_megacomplex, equ_area)

    rela = 1.0  # relation between areas
    irf = dataset_spec.irf
    [sh1, sh2] = dataset_spec.shapes
    pspec_base = {
        "kinetic": [1e-1, 5e-3],
        "i": [0.5, 0.5, {"vary": False}],
        "irf": [["center", irf.location], ["width", irf.width]],
    }
    pspec_equa_area = {
        "rela": [rela, {"vary": False}],
    }
    pspec_shape = {
        "shapes": {
            "amps": [sh1.amplitude, sh2.amplitude],
            "locs": [sh1.location, sh2.location],
            "width": [sh1.width, sh2.width],
        },
    }
    pspec = ParameterSpec(pspec_base, pspec_equa_area, pspec_shape)

    # derivates:
    mspec_sim = dict(deepcopy(mspec.base), **mspec.shape)
    mspec_sim["megacomplex"]["mc2"] = mspec.spectral_megacomplex
    mspec_sim["dataset"]["dataset1"]["global_megacomplex"] = ["mc2"]

    mspec_fit_wp = dict(deepcopy(mspec.base), **mspec.equ_area)
    mspec_fit_np = dict(deepcopy(mspec.base))

    model_sim = SpectralDecayModel.from_dict(mspec_sim)
    model_wp = SpectralDecayModel.from_dict(mspec_fit_wp)
    model_np = SpectralDecayModel.from_dict(mspec_fit_np)
    print(model_np)

    # %% Parameter specification (pspec)

    pspec_sim = dict(deepcopy(pspec.base), **pspec.shapes)
    param_sim = ParameterGroup.from_dict(pspec_sim)

    # For the wp model we create two version of the parameter specification
    # One has all inputs fixed, the other has all but the first free
    # for both we perturb kinetic parameters a bit to give the optimizer some work
    pspec_wp = dict(deepcopy(pspec.base), **pspec.equal_area)
    pspec_wp["kinetic"] = [v * 1.01 for v in pspec_wp["kinetic"]]
    pspec_wp["i"] = [[1, {"vary": False}], 1]
    pspec_np = dict(deepcopy(pspec.base))

    param_wp = ParameterGroup.from_dict(pspec_wp)
    param_np = ParameterGroup.from_dict(pspec_np)

    # %% Print models with parameters
    print(model_sim.markdown(param_sim))
    print(model_wp.markdown(param_wp))
    print(model_np.markdown(param_np))

    # %%
    simulated_data = simulate(
        model_sim,
        "dataset1",
        param_sim,
        coordinates={"time": times, "spectral": wavelengths},
        noise=noise_spec.active,
        noise_std_dev=noise_spec.std_dev,
        noise_seed=noise_spec.seed,
    )
    # %%
    simulated_data = prepare_time_trace_dataset(simulated_data)
    # make a copy to keep an intact reference
    data = deepcopy(simulated_data)

    # %% Optimizing model without penalty (np)

    model_np.dataset_group_models["default"].method = (
        "non_negative_least_squares" if optim_spec.nnls else "variable_projection"
    )

    dataset = {"dataset1": data}
    scheme_np = Scheme(
        model=model_np,
        parameters=param_np,
        data=dataset,
        maximum_number_function_evaluations=optim_spec.max_nfev,
    )
    result_np = optimize(scheme_np)
    print(result_np)

    model_wp.dataset_group_models["default"].method = (
        "non_negative_least_squares" if optim_spec.nnls else "variable_projection"
    )

    # %% Optimizing model with penalty fixed inputs (wp_ifix)
    scheme_wp = Scheme(
        model=model_wp,
        parameters=param_wp,
        data=dataset,
        maximum_number_function_evaluations=optim_spec.max_nfev,
    )
    result_wp = optimize(scheme_wp)
    print(result_wp)

    if debug:
        # %% Plot results
        plt_spec = importlib.util.find_spec("matplotlib")
        if plt_spec is not None:
            import matplotlib.pyplot as plt

            plot_overview(result_np.data["dataset1"], "no penalties")
            plot_overview(result_wp.data["dataset1"], "with penalties")
            plt.show()

    # %% Test calculation
    print(result_wp.data["dataset1"])
    area1_np = np.sum(result_np.data["dataset1"].species_associated_spectra.sel(species="s1"))
    area2_np = np.sum(result_np.data["dataset1"].species_associated_spectra.sel(species="s2"))
    print("area_np", area1_np, area2_np)
    assert not np.isclose(area1_np, area2_np)

    area1_wp = np.sum(result_wp.data["dataset1"].species_associated_spectra.sel(species="s1"))
    area2_wp = np.sum(result_wp.data["dataset1"].species_associated_spectra.sel(species="s2"))
    assert np.isclose(area1_wp, area2_wp)

    input_ratio = result_wp.optimized_parameters.get("i.1") / result_wp.optimized_parameters.get(
        "i.2"
    )
    print("input", input_ratio)
    assert np.isclose(input_ratio, 1.5038858115)


if __name__ == "__main__":
    test_equal_area_penalties(debug=True)
    test_equal_area_penalties(debug=False)
