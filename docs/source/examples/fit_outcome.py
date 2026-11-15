# /// script
# requires-python = ">=3.13"
# dependencies = ["pyglotaran>=0.8"]
# ///
"""Show how a fit can end and how to tell the outcomes apart."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.project import Result

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


def outcome(result: Result) -> str:
    """Classify the outcome of a fit from its optimization info."""
    info = result.optimization_info
    if not info.success:
        return "evaluated only" if info.termination_reason == "Dry run." else "failed"
    if "maximum number of function evaluations" in info.termination_reason:
        return "stopped at the evaluation limit"
    return "converged"


def report(label: str, result: Result) -> None:
    info = result.optimization_info
    rmse = "none" if info.root_mean_square_error is None else f"{info.root_mean_square_error:.4f}"
    print(f"{label}: {outcome(result)}")
    print(
        f"  success={info.success}, evaluations={info.number_of_function_evaluations}, rmse={rmse}"
    )
    print(f"  termination_reason: {info.termination_reason}")


report("fit", scheme.optimize(parameters, {"my_data": dataset}, verbose=False))

report(
    "fit with two evaluations",
    scheme.optimize(
        parameters, {"my_data": dataset}, verbose=False, maximum_number_function_evaluations=2
    ),
)

report("dry run", scheme.optimize(parameters, {"my_data": dataset}, verbose=False, dry_run=True))

# A starting value outside its bounds is an error of the optimizer.
outside = load_parameters("rates: [[1.0, {max: 0.5}], 0.05]", format_name="yml_str")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = scheme.optimize(outside, {"my_data": dataset}, verbose=False)
report("fit with a starting value outside its bounds", result)
print("  warnings:", [str(warning.message).splitlines()[0] for warning in caught])

try:
    scheme.optimize(outside, {"my_data": dataset}, verbose=False, raise_exception=True)
except ValueError as error:
    print("  with raise_exception=True:", type(error).__name__)
