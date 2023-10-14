"""A simple parallel decay for testing purposes."""

from glotaran.builtin.items.activation.gaussian import GaussianActivation
from glotaran.project import Scheme
from glotaran.project.library import ModelLibrary
from glotaran.simulation import simulate
from glotaran.testing.simulated_data.shared_decay import ACTIVATION_BASE
from glotaran.testing.simulated_data.shared_decay import LIBRARY
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
from glotaran.testing.simulated_data.shared_decay import SIMULATION_COORDINATES
from glotaran.testing.simulated_data.shared_decay import SIMULATION_PARAMETERS
from glotaran.testing.simulated_data.shared_decay import KineticSpectrumDataModel

ACTIVATION = ACTIVATION_BASE | {"compartments": {"s1": 1, "s2": 1}}

DATASET = simulate(
    KineticSpectrumDataModel(
        elements=["parallel"],
        global_elements=["spectral"],
        activation=[GaussianActivation.parse_obj(ACTIVATION)],  # type:ignore[call-arg]
    ),
    ModelLibrary.from_dict(LIBRARY),
    SIMULATION_PARAMETERS,
    SIMULATION_COORDINATES,
    noise=True,
    noise_std_dev=1e-2,
)

SCHEME_DICT = {
    "library": LIBRARY,
    "experiments": {
        "parallel-decay": {
            "datasets": {
                "parallel-decay": {
                    "elements": ["parallel"],
                    "activation": [ACTIVATION],
                }
            }
        }
    },
}

SCHEME = Scheme.from_dict(SCHEME_DICT)
SCHEME.load_data({"parallel-decay": DATASET})

RESULT = SCHEME.optimize(PARAMETERS)
