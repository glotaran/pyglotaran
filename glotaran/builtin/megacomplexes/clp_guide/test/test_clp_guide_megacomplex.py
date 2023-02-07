import numpy as np

from glotaran.builtin.megacomplexes.clp_guide import ClpGuideMegacomplex
from glotaran.builtin.megacomplexes.decay import DecaySequentialMegacomplex
from glotaran.builtin.megacomplexes.decay.test.test_decay_megacomplex import create_gaussian_clp
from glotaran.model import Model
from glotaran.optimization.optimize import optimize
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation.simulation import simulate


def test_clp_guide():
    model = Model.create_class_from_megacomplexes(
        [DecaySequentialMegacomplex, ClpGuideMegacomplex]
    )(
        **{
            "dataset_groups": {"default": {"link_clp": True}},
            "megacomplex": {
                "mc1": {
                    "type": "decay-sequential",
                    "compartments": ["s1", "s2"],
                    "rates": ["1", "2"],
                },
                "mc2": {"type": "clp-guide", "dimension": "time", "target": "s1"},
            },
            "dataset": {
                "dataset1": {"megacomplex": ["mc1"]},
                "dataset2": {"megacomplex": ["mc2"]},
            },
        },
    )

    initial_parameters = Parameters.from_list(
        [101e-5, 501e-4, [1, {"vary": False, "non-negative": False}]]
    )
    wanted_parameters = Parameters.from_list(
        [101e-4, 501e-3, [1, {"vary": False, "non-negative": False}]]
    )

    time = np.arange(0, 50, 1.5)
    pixel = np.arange(600, 750, 5)
    axis = {"time": time, "pixel": pixel}

    clp = create_gaussian_clp(["s1", "s2"], [7, 30], [620, 720], [10, 50], pixel)

    dataset1 = simulate(model, "dataset1", wanted_parameters, axis, clp)
    dataset2 = clp.sel(clp_label=["s1"]).rename(clp_label="time")
    data = {"dataset1": dataset1, "dataset2": dataset2}

    scheme = Scheme(
        model=model,
        parameters=initial_parameters,
        data=data,
        maximum_number_function_evaluations=20,
    )
    result = optimize(scheme)
    print(result.optimized_parameters)
    for param in result.optimized_parameters.all():
        assert np.allclose(param.value, wanted_parameters.get(param.label).value, rtol=1e-1)
