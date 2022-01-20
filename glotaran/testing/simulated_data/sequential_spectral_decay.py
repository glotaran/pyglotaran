"""A simple sequential decay for testing purposes."""

from glotaran.analysis.simulation import simulate
from glotaran.io import load_model
from glotaran.project import Scheme
from glotaran.project.generators import generate_model_yml
from glotaran.testing.simulated_data.shared_decay import PARAMETER
from glotaran.testing.simulated_data.shared_decay import SIMULATION_COORDINATES
from glotaran.testing.simulated_data.shared_decay import SIMULATION_PARAMETER
from glotaran.testing.simulated_data.shared_decay import *  # noqa F403

SIMULATION_MODEL_YML = generate_model_yml(
    generator_name="spectral_decay_sequential",
    generator_arguments={"nr_compartments": 3, "irf": True},  # type:ignore[arg-type]
)
SIMULATION_MODEL = load_model(SIMULATION_MODEL_YML, format_name="yml_str")

MODEL_YML = generate_model_yml(
    generator_name="decay_sequential",
    generator_arguments={"nr_compartments": 3, "irf": True},  # type:ignore[arg-type]
)
MODEL = load_model(MODEL_YML, format_name="yml_str")


DATASET = simulate(
    SIMULATION_MODEL,
    "dataset_1",
    SIMULATION_PARAMETER,
    SIMULATION_COORDINATES,
    noise=True,
    noise_std_dev=1e-2,
)

SCHEME = Scheme(model=MODEL, parameters=PARAMETER, data={"dataset_1": DATASET})
