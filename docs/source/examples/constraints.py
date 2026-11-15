# /// script
# requires-python = ">=3.13"
# dependencies = ["pyglotaran>=0.8"]
# ///
"""Constrain the spectra of a fit: a zero constraint, a relation and an equal area penalty."""

from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.io import load_parameters
from glotaran.io import load_scheme

# Two species with the rates 0.6 and 0.1. The spectrum of s1 is zero above 660,
# and up to 612 the spectrum of s2 is half the spectrum of s1.
time = np.linspace(0, 20, 201)
spectral = np.linspace(600, 700, 51)
spectrum_s1 = np.exp(-(((spectral - 630) / 12) ** 2)) * (spectral <= 660)
spectrum_s2 = np.exp(-(((spectral - 670) / 12) ** 2)) + 0.5 * spectrum_s1 * (spectral <= 612)
signal = np.outer(np.exp(-0.6 * time), spectrum_s1) + np.outer(np.exp(-0.1 * time), spectrum_s2)
noise = np.random.default_rng(seed=0).normal(scale=0.01, size=signal.shape)
dataset = xr.Dataset(
    {"data": (("time", "spectral"), signal + noise)},
    coords={"time": time, "spectral": spectral},
)
area_ratio = spectrum_s1.sum() / spectrum_s2.sum()

SCHEME = """
library:
  decay:
    type: kinetic
    rates:
      (s1, s1): rates.1
      (s2, s2): rates.2
{clp_constraints}
experiments:
  my_experiment:
    datasets:
      my_data:
        elements: [decay]
        activations:
          excitation:
            type: instant
            compartments: {{s1: 1, s2: 1}}
{experiment_fields}
"""
parameters = load_parameters("rates: [1.0, 0.05]", format_name="yml_str")


def fit(clp_constraints: str = "", experiment_fields: str = "") -> xr.DataArray:
    """Fit the dataset and return the spectra of the two species."""
    scheme = load_scheme(
        SCHEME.format(clp_constraints=clp_constraints, experiment_fields=experiment_fields),
        format_name="yml_str",
    )
    result = scheme.optimize(parameters, {"my_data": dataset}, verbose=False)
    rates = [round(float(p.value), 3) for p in result.optimized_parameters.all()]
    print(f"  rates: {rates}")
    return result.optimization_results["my_data"].elements["decay"].amplitudes


def report(spectra: xr.DataArray) -> None:
    s1, s2 = spectra.sel(compartment="s1"), spectra.sel(compartment="s2")
    print(
        f"  largest amplitude of s1 above 660: {float(abs(s1.sel(spectral=slice(661, 700))).max()):.4f}"
    )
    ratio = s2.sel(spectral=slice(600, 612)) / s1.sel(spectral=slice(600, 612))
    print(f"  s2 / s1 up to 612: {float(ratio.min()):.3f} to {float(ratio.max()):.3f}")
    print(f"  area of s1 / area of s2: {float(s1.sum() / s2.sum()):.3f}")


print(f"simulated: s1 is zero above 660, s2 / s1 is 0.5 up to 612, area ratio {area_ratio:.3f}")

print("without constraints")
report(fit())

print("zero constraint")
report(
    fit(
        clp_constraints="""
    clp_constraints:
      - type: zero
        target: s1
        interval: [[661, 700]]
"""
    )
)

print("relation")
report(
    fit(
        experiment_fields="""
    clp_relations:
      - source: s1
        target: s2
        parameter: 0.5
        interval: [[600, 612]]
"""
    )
)

print("equal area penalty")
report(
    fit(
        experiment_fields=f"""
    clp_penalties:
      - type: equal_area
        source: s1
        target: s2
        parameter: {area_ratio:.4f}
        weight: 0.1
"""
    )
)
