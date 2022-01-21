"""Shared variables for simulated decays."""
import numpy as np

from glotaran.io import load_parameters

SIMULATION_PARAMETER_YML = """
rates:
  - [species_1, 0.5]
  - [species_2, 0.3]
  - [species_3, 0.1]

irf:
  - [center, 0.3]
  - [width, 0.1]

shapes:
  species_1:
    - [amplitude, 30]
    - [location, 620]
    - [width, 40]
  species_2:
    - [amplitude, 20]
    - [location, 630]
    - [width, 20]
  species_3:
    - [amplitude, 60]
    - [location, 650]
    - [width, 60]
"""
SIMULATION_PARAMETER = load_parameters(SIMULATION_PARAMETER_YML, format_name="yml_str")

PARAMETER_YML = """
rates:
  - [species_1, 0.5]
  - [species_2, 0.3]
  - [species_3, 0.1]

irf:
  - [center, 0.3]
  - [width, 0.1]
"""
PARAMETER = load_parameters(PARAMETER_YML, format_name="yml_str")

TIME_AXIS = np.arange(-1, 20, 0.01)
SPECTRAL_AXIS = np.arange(600, 700, 1.4)
SIMULATION_COORDINATES = {"time": TIME_AXIS, "spectral": SPECTRAL_AXIS}
