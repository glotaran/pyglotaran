"""A simple parallel decay for testing purposes."""

from glotaran.io import load_model
from glotaran.project import Scheme
from glotaran.project.generators import generate_model_yml
from glotaran.simulation import simulate
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
from glotaran.testing.simulated_data.shared_decay import SIMULATION_COORDINATES
from glotaran.testing.simulated_data.shared_decay import SIMULATION_PARAMETERS

SIMULATION_MODEL_YML = generate_model_yml(
    generator_name="spectral_decay_parallel",
    generator_arguments={"nr_compartments": 3, "irf": True},
)
SIMULATION_MODEL = load_model(SIMULATION_MODEL_YML, format_name="yml_str")

MODEL_YML = generate_model_yml(
    generator_name="decay_parallel",
    generator_arguments={"nr_compartments": 3, "irf": True},
)
MODEL = load_model(MODEL_YML, format_name="yml_str")

DATASET = simulate(
    SIMULATION_MODEL,
    "dataset_1",
    SIMULATION_PARAMETERS,
    SIMULATION_COORDINATES,
    noise=True,
    noise_std_dev=1e-2,
)

SCHEME = Scheme(model=MODEL, parameters=PARAMETERS, data={"dataset_1": DATASET})
