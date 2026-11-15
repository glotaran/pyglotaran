# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = ["pyglotaran>=0.8"]
# ///
"""Fit a simulated dataset with pyglotaran, save the result and load it again."""

from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.io import load_parameters
from glotaran.io import load_result
from glotaran.io import load_scheme
from glotaran.io import save_result

# 1. Data: two species that decay with the rates 0.6 and 0.1, each with its own spectrum.
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

# 2. Scheme: the model. `load_scheme("scheme.yml")` reads the same text from a file.
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
            compartments:
              s1: 1
              s2: 1
""",
    format_name="yml_str",
)

# 3. Parameters: the starting values, with the labels `rates.1` and `rates.2`.
parameters = load_parameters("rates: [1.0, 0.05]", format_name="yml_str")

# 4. Fit.
result = scheme.optimize(parameters, {"my_data": dataset}, verbose=False)

info = result.optimization_info
print("success:", info.success)
print("termination:", info.termination_reason)
print(f"root mean square error: {info.root_mean_square_error:.4f}")
for parameter in result.optimized_parameters.all():
    print(f"{parameter.label} = {parameter.value:.4f} +/- {parameter.standard_error:.4f}")

# 5. Save the result to a folder and load it again.
save_result(result, "quickstart_result/result.yml", allow_overwrite=True)
loaded = load_result("quickstart_result")

fit = loaded.optimization_results["my_data"]
print("loaded rates:", [round(p.value, 4) for p in loaded.optimized_parameters.all()])
print("arrays of the element 'decay':", sorted(fit.elements["decay"].data_vars))
print("shape of the fitted data:", fit.fitted_data.shape)
