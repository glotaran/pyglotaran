"""Shared variables for simulated decays."""

from __future__ import annotations

import numpy as np

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.elements.spectral import SpectralElement
from glotaran.io import load_parameters
from glotaran.model.data_model import DataModel

SIMULATION_PARAMETERS_YML = """
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
SIMULATION_PARAMETERS = load_parameters(SIMULATION_PARAMETERS_YML, format_name="yml_str")

PARAMETERS_YML = """
rates:
  - [species_1, 0.5]
  - [species_2, 0.3]
  - [species_3, 0.1]

irf:
  - [center, 0.3]
  - [width, 0.1]
"""
PARAMETERS = load_parameters(PARAMETERS_YML, format_name="yml_str")

TIME_AXIS = np.arange(-1, 20, 0.01)
SPECTRAL_AXIS = np.arange(600, 700, 1.4)
SIMULATION_COORDINATES = {"time": TIME_AXIS, "spectral": SPECTRAL_AXIS}

KineticSpectrumDataModel = DataModel.create_class_for_elements({KineticElement, SpectralElement})

LIBRARY = {
    "sequential": {
        "type": "kinetic",
        "rates": {
            ("s2", "s1"): "rates.species_1",
            ("s3", "s2"): "rates.species_2",
            ("s3", "s3"): "rates.species_3",
        },
    },
    "parallel": {
        "type": "kinetic",
        "rates": {
            ("s1", "s1"): "rates.species_1",
            ("s2", "s2"): "rates.species_2",
            ("s3", "s3"): "rates.species_3",
        },
    },
    "spectral": {
        "type": "spectral",
        "shapes": {
            "s1": {
                "type": "gaussian",
                "amplitude": "shapes.species_1.amplitude",
                "location": "shapes.species_1.location",
                "width": "shapes.species_1.width",
            },
            "s2": {
                "type": "gaussian",
                "amplitude": "shapes.species_2.amplitude",
                "location": "shapes.species_2.location",
                "width": "shapes.species_2.width",
            },
            "s3": {
                "type": "gaussian",
                "amplitude": "shapes.species_3.amplitude",
                "location": "shapes.species_3.location",
                "width": "shapes.species_3.width",
            },
        },
    },
}

ACTIVATION_BASE = {
    "type": "gaussian",
    "center": "irf.center",
    "width": "irf.width",
}
