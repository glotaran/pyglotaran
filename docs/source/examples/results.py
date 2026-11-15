# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = ["pyglotaran>=0.8"]
# ///
"""Read the result of a fit, save it, and read the saved files with and without pyglotaran."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from glotaran.io import SAVING_OPTIONS_MINIMAL
from glotaran.io import load_parameters
from glotaran.io import load_result
from glotaran.io import load_scheme

time = np.linspace(0, 20, 201)
spectral = np.linspace(600, 700, 31)
decays = [np.exp(-rate * time) for rate in (0.6, 0.1)]
spectra = [np.exp(-(((spectral - center) / 15) ** 2)) for center in (630, 670)]
signal = sum(np.outer(decay, spectrum) for decay, spectrum in zip(decays, spectra, strict=True))
noise = np.random.default_rng(seed=0).normal(scale=0.01, size=signal.shape)
dataset = xr.Dataset(
    {"data": (("time", "spectral"), signal + noise)},
    coords={"time": time, "spectral": spectral},
)
scheme = load_scheme(
    """
library:
  decay:
    type: kinetic
    rates:
      (s1, s1): rates.1
      (s2, s2): rates.2

experiments:
  my_experiment:
    datasets:
      my_data:
        elements: [decay]
        activations:
          excitation:
            type: instant
            compartments: {s1: 1, s2: 1}
""",
    format_name="yml_str",
)
parameters = load_parameters("rates: [1.0, 0.05]", format_name="yml_str")
result = scheme.optimize(parameters, {"my_data": dataset}, verbose=False)

# Parameters: a table with values and standard errors.
table = result.optimized_parameters.to_dataframe()
print(table[["label", "value", "standard_error"]].round(4).to_string(index=False))

# Correlation of the free parameters, from the covariance matrix.
info = result.optimization_info
errors = np.sqrt(np.diag(info.covariance_matrix))
correlation = info.covariance_matrix / np.outer(errors, errors)
print("free parameters:", info.free_parameter_labels)
print("correlation of the two rates:", round(float(correlation[0, 1]), 3))

# Arrays of one dataset.
fit = result.optimization_results["my_data"]
decay = fit.elements["decay"]
print("spectra:", dict(decay.amplitudes.sizes))
print("concentrations:", dict(decay.concentrations.sizes))
print(
    "residuals:", dict(fit.residuals.sizes), "largest:", round(float(abs(fit.residuals).max()), 4)
)
print("meta:", fit.meta.model_dimension, fit.meta.global_dimension)

# Save everything, or only the parameters and the statistics.
result.save("results_example", allow_overwrite=True)
result.save("results_example_minimal", saving_options=SAVING_OPTIONS_MINIMAL, allow_overwrite=True)

# Load with pyglotaran.
loaded = load_result("results_example")
print("loaded:", sorted(loaded.optimization_results["my_data"].elements["decay"].data_vars))

# Read the files with other tools: NetCDF for arrays, CSV for parameters.
saved_spectra = xr.open_dataset("results_example/optimization_results/my_data/elements/decay.nc")
saved_parameters = pd.read_csv("results_example/optimized_parameters.csv")
print("decay.nc:", dict(saved_spectra.amplitudes.sizes))
print("optimized_parameters.csv:", list(saved_parameters.columns))
