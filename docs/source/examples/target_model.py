# /// script
# requires-python = ">=3.13"
# dependencies = ["pyglotaran>=0.8"]
# ///
"""Simulate a sequential model with an instrument response function and fit it."""

from __future__ import annotations

import numpy as np

from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.simulation import simulate

# The model: s1 is excited, converts to s2 with rates.1, and s2 decays with rates.2.
MODEL = """
library:
  sequential:
    type: kinetic
    rates:
      (s2, s1): rates.1
      (s2, s2): rates.2
{spectra}
experiments:
  my_experiment:
    datasets:
      my_data:
        elements: [sequential]
{global_elements}
        activations:
          irf:
            type: gaussian
            center: irf.center
            width: irf.width
            compartments:
              s1: 1
"""

# For the simulation, a spectral element gives each compartment a spectrum.
simulation_scheme = load_scheme(
    MODEL.format(
        spectra="""
  spectra:
    type: spectral
    shapes:
      s1: {type: gaussian, amplitude: 1, location: 620, width: 20}
      s2: {type: gaussian, amplitude: 1, location: 660, width: 25}
""",
        global_elements="        global_elements: [spectra]",
    ),
    format_name="yml_str",
)
true_parameters = load_parameters(
    """
rates: [2.0, 0.1]
irf:
  - [center, 0.5]
  - [width, 0.1]
""",
    format_name="yml_str",
)
dataset = simulate(
    simulation_scheme.experiments["my_experiment"].datasets["my_data"],
    simulation_scheme.library,
    true_parameters,
    {"time": np.arange(-1, 30, 0.05), "spectral": np.arange(580, 720, 4.0)},
    noise=True,
    noise_std_dev=0.005,
)

# For the fit, the spectra are free: every wavelength gets its own amplitudes.
scheme = load_scheme(MODEL.format(spectra="", global_elements=""), format_name="yml_str")
parameters = load_parameters(
    """
rates:
  - [1.0, {min: 0}]
  - [0.2, {min: 0}]
irf:
  - [center, 0.4]
  - [width, 0.2, {min: 0.01}]
""",
    format_name="yml_str",
)
result = scheme.optimize(parameters, {"my_data": dataset}, verbose=False)

print("termination:", result.optimization_info.termination_reason)
for parameter in result.optimized_parameters.all():
    print(f"{parameter.label} = {parameter.value:.4f} +/- {parameter.standard_error:.4f}")

sequential = result.optimization_results["my_data"].elements["sequential"]
print("species associated spectra:", dict(sequential.amplitudes.sizes))
peak = sequential.amplitudes.idxmax("spectral")
print(
    "spectral maxima:", {str(c.item()): float(peak.sel(compartment=c)) for c in peak.compartment}
)
