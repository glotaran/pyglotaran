# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.parameter import ParameterGroup
from glotaran.io import prepare_time_trace_dataset

from pyglotaran_examples.io.load_data import load_data
from pyglotaran_examples.plotting.style import PlotStyle
from pyglotaran_examples.plotting.plot_overview import plot_overview

plot_style = PlotStyle()
plt.rc("axes", prop_cycle=plot_style.cycler)

# %%
weight = 0.1
rel1 = 1
pen = 0.05
wavelengths = np.arange(650, 670, 1)
times = np.asarray(np.arange(-1, 99, 0.1))
time_p1 = np.linspace(-1, 2, 50, endpoint=False)
time_p2 = np.linspace(2, 10, 30, endpoint=False)
time_p3 = np.geomspace(10, 50, num=20)
time = np.concatenate([time_p1, time_p2, time_p3])
wavelengths = np.linspace(650, 670, 20, endpoint=True)

optim_arg_max_nfev = 99
loc_sh12 = float(wavelengths[5])  # 2
loc_sh3 = float(wavelengths[5])  # -3
amp_sh12 = 1
amp_sh3 = 1
shape_width = float((wavelengths[1] - wavelengths[0]) * 3)
irf_loc = float(times[20])
equ_interval = [(min(wavelengths), max(wavelengths))]
irf_width = float((times[1] - times[0]) * 10)


def calculate_gaussian(axis, amplitude, location, width):
    matrix = amplitude * np.exp(-np.log(2) * np.square(2 * (axis - location) / width))
    return matrix


# %% The base model used for simulation
sim_model_dict = {
    "initial_concentration": {
        "j1": {
            "compartments": ["s1", "s2", "s3"],
            "parameters": ["i.1", "i.2", "i.3"],
        },
    },
    "shape": {
        "sh1": {
            "type": "gaussian",
            "amplitude": "shape.amps.1",
            "location": "shape.locs.1",
            "width": "shape.width.1",
        },
        "sh2": {
            "type": "gaussian",
            "amplitude": "shape.amps.2",
            "location": "shape.locs.2",
            "width": "shape.width.2",
        },
        "sh3": {
            "type": "gaussian",
            "amplitude": "shape.amps.3",
            "location": "shape.locs.3",
            "width": "shape.width.3",
        },
    },
    "megacomplex": {
        "mc1": {"k_matrix": ["k1"]},
    },
    "k_matrix": {
        "k1": {
            "matrix": {
                ("s1", "s1"): "kinetic.1",
                ("s2", "s2"): "kinetic.2",
                ("s3", "s3"): "kinetic.3",
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
            "shape": {
                "s1": "sh1",
                "s2": "sh2",
                "s3": "sh3",
            },
            "irf": "irf1",
        },
    },
}

penalties_dict = {
    "equal_area_penalties": [
        {
            "compartment": "s1",
            "target": "s3",
            "parameter": "pen.1",
            "interval": equ_interval,
            "weight": weight,
        },
    ],
    "spectral_relations": [
        {
            "compartment": "s1",
            "target": "s2",
            "parameter": "rel.1",
            "interval": equ_interval,
        },
    ],
}

# derivates:
fit_model_dict = deepcopy(sim_model_dict)
del fit_model_dict["shape"]
del fit_model_dict["dataset"]["dataset1"]["shape"]

model_wp_dict = dict(fit_model_dict, **penalties_dict)
model_np_dict = deepcopy(fit_model_dict)

simulation_model = KineticSpectrumModel.from_dict(sim_model_dict)
model_wp = KineticSpectrumModel.from_dict(model_wp_dict)
model_np = KineticSpectrumModel.from_dict(model_np_dict)

# %%

base_param_dict = {
    "kinetic": [1e-1, 5e-3, 3e-2],
    "i": [1, 1, 1, {'vary': False}],
    "rel": [rel1, {'vary': False}],
    "pen": [pen, {'vary': False}],
    "irf": [["center", irf_loc], ["width", irf_width]],
    "shape": {
        "amps": [amp_sh12, amp_sh12, amp_sh3],
        "locs": [loc_sh12, loc_sh12, loc_sh3],
        "width": [shape_width, shape_width, shape_width],
    },
}

sim_param_dict = deepcopy(base_param_dict)
del sim_param_dict["rel"]
del sim_param_dict["pen"]
simulation_parameters = ParameterGroup.from_dict(sim_param_dict)
print(simulation_parameters)

fit_param_wp_dict = deepcopy(base_param_dict)
del fit_param_wp_dict["shape"]
fit_param_wp_dict["kinetic"] = [v * 1.02 for v in fit_param_wp_dict["kinetic"]]

fit_param_np_dict = deepcopy(fit_param_wp_dict)
del fit_param_np_dict["pen"]

fit_parameters_wp = ParameterGroup.from_dict(fit_param_wp_dict)
print(fit_parameters_wp)
fit_parameters_np = ParameterGroup.from_dict(fit_param_np_dict)
print(fit_parameters_np)

# %% Print models with parameters
print(simulation_model.markdown(simulation_parameters))
print(model_wp.markdown(fit_parameters_wp))
print(model_np.markdown(fit_parameters_np))

# %%
simulated_data = simulation_model.simulate(
    "dataset1",
    simulation_parameters,
    axes={"time": times, "spectral": wavelengths},
)
# %%
simulated_data = prepare_time_trace_dataset(simulated_data)
simulated_data.data.plot()
# make a copy to keep an intact reference
data = deepcopy(simulated_data)

# %%
# TODO: check data
data.data.plot()
plt.show()

# %%
result_wp = model_wp.optimize(
    fit_parameters_wp, {"dataset1": data}, nnls=True, max_nfev=optim_arg_max_nfev
)
print(result_wp)

# %%
# TODO: check results
res_with_penalty = load_data(result_wp)
plot_overview(result_wp, linlog=False)  # linrange=(-1,1)
plt.show()

# %% Optimizing model without penalty


result_np = model_np.optimize(
    fit_parameters_np, {"dataset1": data}, nnls=True, max_nfev=optim_arg_max_nfev
)
print(result_np)

# %%
# TODO: check results
plot_overview(result_np, linlog=False)

# %% Test calculation
result_data = result_wp.data["dataset1"]
wanted_penalty = (
    np.sum(result_data.species_associated_spectra.sel(species="s2"))
    - np.sum(result_data.species_associated_spectra.sel(species="s3")) * pen
)
wanted_penalty *= weight
wanted_penalty **= 2
wanted_penalty = np.sum(wanted_penalty.values)

additional_penalty = result_wp.chisqr - result_np.chisqr
print(f"additional_penalty: {additional_penalty}")
print(f"wanted_penalty: {wanted_penalty}")
print(f"diff:{np.isclose(additional_penalty, wanted_penalty)}")


# %%


# %%


# %%
