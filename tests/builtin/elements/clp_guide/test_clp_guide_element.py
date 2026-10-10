from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.builtin.elements.clp_guide import ClpGuideElement
from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import InstantActivation
from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.optimization import OptimizationData
from glotaran.optimization import OptimizationMatrix
from glotaran.parameter import Parameters
from glotaran.simulation import simulate


def test_clp_guide():
    model = DataModel(
        data=xr.DataArray(np.ones((1, 1)), coords=[("model", [0]), ("global", [0])]).to_dataset(
            name="data"
        ),
        elements=[ClpGuideElement(type="clp-guide", label="test", target="c", dimension="model")],
    )
    data = OptimizationData(model)

    matrix = OptimizationMatrix.from_data(data)

    assert len(matrix.clp_axis) == 1
    assert ("c") in matrix.clp_axis

    assert matrix.array.shape == (1, 1)
    assert np.all(matrix.array[0, 0] == 1)


def test_clp_guide_linked_fit():
    library = {
        "sequential": KineticElement(
            label="sequential",
            type="kinetic",
            rates={("s2", "s1"): "rates.1", ("s2", "s2"): "rates.2"},
        ),
        "guide": ClpGuideElement(label="guide", type="clp-guide", target="s1", dimension="time"),
    }
    wanted_parameters = Parameters.from_dict({"rates": [101e-4, 501e-3]})
    initial_parameters = Parameters.from_dict({"rates": [101e-5, 501e-4]})

    pixel = np.arange(600, 750, 5)
    clp = xr.DataArray(
        [
            7 * np.exp(-np.log(2) * np.square(2 * (pixel - 620) / 10)),
            30 * np.exp(-np.log(2) * np.square(2 * (pixel - 720) / 50)),
        ],
        coords=[("clp_label", ["s1", "s2"]), ("pixel", pixel)],
    ).T

    decay_model = ActivationDataModel(
        elements=["sequential"],
        activations={"irf": InstantActivation(type="instant", compartments={"s1": 1})},
    )
    decay_model.data = simulate(
        decay_model,
        library,
        wanted_parameters,
        {"time": np.arange(0, 50, 1.5), "pixel": pixel},
        clp=clp,
    )
    guide_data = clp.sel(clp_label=["s1"]).rename(clp_label="time").assign_coords(time=[0])
    guide_model = DataModel(elements=["guide"], data=guide_data.to_dataset(name="data"))

    optimization = Optimization(
        models=[ExperimentModel(datasets={"decay": decay_model, "guide": guide_model})],
        parameters=initial_parameters,
        library=library,
        raise_exception=True,
        maximum_number_function_evaluations=20,
    )
    optimized_parameters, optimized_data, _ = optimization.run()

    for parameter in optimized_parameters.all():
        assert np.allclose(
            parameter.value, wanted_parameters.get(parameter.label).value, rtol=1e-1
        )

    guide_result = optimized_data["guide"]
    assert guide_result.fit_decomposition is not None
    assert np.allclose(
        guide_result.fit_decomposition.clp.sel(amplitude_label="s1"), clp.sel(clp_label="s1")
    )
    assert len(guide_result.elements["guide"].data_vars) == 0
    assert "element_uid" in guide_result.elements["guide"].attrs
