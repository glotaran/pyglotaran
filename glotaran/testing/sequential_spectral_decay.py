"""A simple sequential decay for testing purposes."""
import numpy as np

from glotaran.analysis.simulation import simulate
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.project import Scheme
from glotaran.project.generators import generate_model_yml

SIMULATION_MODEL_YML = generate_model_yml(
    "spectral-decay-sequential", **{"nr_compartments": 3, "irf": True}
)
SIMULATION_MODEL = load_model(SIMULATION_MODEL_YML, format_name="yml_str")

MODEL_YML = generate_model_yml("decay-sequential", **{"nr_compartments": 3, "irf": True})
MODEL = load_model(MODEL_YML, format_name="yml_str")

WANTED_PARAMETER_YML = """
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
WANTED_PARAMETER = load_parameters(WANTED_PARAMETER_YML, format_name="yml_str")

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

DATASET = simulate(
    SIMULATION_MODEL,
    "dataset_1",
    WANTED_PARAMETER,
    {"time": TIME_AXIS, "spectral": SPECTRAL_AXIS},
    noise=True,
    noise_std_dev=1e-2,
)

SCHEME = Scheme(model=MODEL, parameters=PARAMETER, data={"dataset_1": DATASET})
