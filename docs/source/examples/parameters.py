# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = ["pyglotaran>=0.8"]
# ///
"""Define parameters with options, change them in code, and save them as a table."""

from __future__ import annotations

from glotaran.io import load_parameters
from glotaran.io import save_parameters

parameters = load_parameters(
    """
rates:
  - [fast, 2.0, {min: 0}]
  - [slow, 0.1, {non-negative: true}]
  - [total, 0, {expr: "$rates.fast + $rates.slow"}]

irf:
  - [center, 0.5, {vary: false}]
  - [width, 0.1, {min: 0.01, max: 1}]

amplitudes: [1.0, 0.5]
""",
    format_name="yml_str",
)

columns = ["label", "value", "expression", "minimum", "maximum", "non_negative", "vary"]
print(parameters.to_dataframe()[columns].to_string(index=False))

# One parameter, by its label.
width = parameters.get("irf.width")
print("irf.width:", width.value, width.minimum, width.maximum, width.vary)

# Change a starting value or fix a parameter before a fit.
parameters.get("rates.fast").value = 3.0
parameters.get("amplitudes.2").vary = False

# A table file keeps every field; YAML can be loaded but not saved.
save_parameters(parameters, "parameters_example.csv", allow_overwrite=True)
reloaded = load_parameters("parameters_example.csv")
print("reloaded labels:", reloaded.labels)
print("rates.fast:", reloaded.get("rates.fast").value)
print("amplitudes.2 varies:", reloaded.get("amplitudes.2").vary)
print(
    "rates.total:", reloaded.get("rates.total").value, "=", reloaded.get("rates.total").expression
)
