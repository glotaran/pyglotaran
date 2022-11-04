import numpy as np

from glotaran.optimization.optimize import optimize
from glotaran.optimization.test.models import DecayModel
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation import simulate


def test_multiple_groups():
    wanted_parameters = Parameters.from_list([101e-4])
    initial_parameters = Parameters.from_list([100e-5])

    global_axis = np.asarray([1.0])
    model_axis = np.arange(0, 150, 1.5)

    sim_model_dict = {
        "megacomplex": {
            "m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False},
            "m2": {"type": "simple-spectral-test-mc"},
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ["m1"],
                "global_megacomplex": ["m2"],
                "kinetic": ["1"],
            }
        },
    }
    sim_model = DecayModel(**sim_model_dict)
    model_dict = {
        "dataset_groups": {"g1": {}, "g2": {"residual_function": "non_negative_least_squares"}},
        "megacomplex": {"m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False}},
        "dataset": {
            "dataset1": {
                "group": "g1",
                "megacomplex": ["m1"],
                "kinetic": ["1"],
            },
            "dataset2": {
                "group": "g2",
                "megacomplex": ["m1"],
                "kinetic": ["1"],
            },
        },
    }
    model = DecayModel(**model_dict)
    dataset = simulate(
        sim_model,
        "dataset1",
        wanted_parameters,
        {"global": global_axis, "model": model_axis},
    )
    scheme = Scheme(
        model=model,
        parameters=initial_parameters,
        data={"dataset1": dataset, "dataset2": dataset},
        maximum_number_function_evaluations=10,
        clp_link_tolerance=0.1,
    )

    result = optimize(scheme, raise_exception=True)
    print(result.optimized_parameters)
    assert result.success
    for param in result.optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, wanted_parameters.get(param.label).value, rtol=1e-1)

    for dataset in result.data.values():
        assert "weighted_root_mean_square_error" in dataset.attrs
        assert "fitted_data" in dataset.data_vars
