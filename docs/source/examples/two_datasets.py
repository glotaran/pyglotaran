# /// script
# requires-python = ">=3.13"
# dependencies = ["pyglotaran>=0.8"]
# ///
"""Fit two datasets of the same sample together, with shared rates and spectra."""

from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.io import load_parameters
from glotaran.io import load_scheme

spectral = np.linspace(600, 700, 31)
spectra = [np.exp(-(((spectral - center) / 15) ** 2)) for center in (630, 670)]


def measure(time: np.ndarray, amplitude: float, seed: int) -> xr.Dataset:
    """Create a dataset of two species that decay with the rates 0.6 and 0.1."""
    decays = [np.exp(-rate * time) for rate in (0.6, 0.1)]
    signal = amplitude * sum(np.outer(d, s) for d, s in zip(decays, spectra, strict=True))
    noise = np.random.default_rng(seed).normal(scale=0.01, size=signal.shape)
    return xr.Dataset(
        {"data": (("time", "spectral"), signal + noise)},
        coords={"time": time, "spectral": spectral},
    )


# The second measurement has a longer time range and half the signal.
datasets = {
    "short": measure(np.linspace(0, 5, 101), amplitude=1.0, seed=0),
    "long": measure(np.linspace(0, 40, 201), amplitude=0.5, seed=1),
}

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
    scale:
      long: scale.long
    datasets:
      short:
        elements: [decay]
        activations:
          excitation:
            type: instant
            compartments: {s1: 1, s2: 1}
      long:
        elements: [decay]
        activations:
          excitation:
            type: instant
            compartments: {s1: 1, s2: 1}
""",
    format_name="yml_str",
)
parameters = load_parameters(
    """
rates: [1.0, 0.05]
scale:
  - [long, 1.0]
""",
    format_name="yml_str",
)

result = scheme.optimize(parameters, datasets, verbose=False)

print("termination:", result.optimization_info.termination_reason)
for parameter in result.optimized_parameters.all():
    print(f"{parameter.label} = {parameter.value:.4f} +/- {parameter.standard_error:.4f}")

for label, fit in result.optimization_results.items():
    print(f"{label}: root mean square error {fit.meta.root_mean_square_error:.4f}")

short = result.optimization_results["short"].elements["decay"].amplitudes
long = result.optimization_results["long"].elements["decay"].amplitudes
print("same spectra in both datasets:", bool(np.allclose(short, long)))
