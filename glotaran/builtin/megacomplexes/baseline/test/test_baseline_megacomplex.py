import numpy as np

from glotaran.builtin.megacomplexes.baseline import BaselineMegacomplex
from glotaran.model import Model
from glotaran.model import fill_item
from glotaran.optimization.matrix_provider import MatrixProvider
from glotaran.parameter import Parameters


def test_baseline():
    model = Model.create_class_from_megacomplexes([BaselineMegacomplex])(
        **{
            "megacomplex": {"m": {"type": "baseline", "dimension": "time"}},
            "dataset": {"dataset1": {"megacomplex": ["m"]}},
        }
    )

    parameters = Parameters({})
    time = np.asarray(np.arange(0, 50, 1.5))
    pixel = np.asarray([0])
    dataset_model = fill_item(model.dataset["dataset1"], model, parameters)
    matrix = MatrixProvider.calculate_dataset_matrix(dataset_model, pixel, time)
    compartments = matrix.clp_labels

    assert len(compartments) == 1
    assert "dataset1_baseline" in compartments

    assert matrix.matrix.shape == (time.size, 1)
    assert np.all(matrix.matrix[:, 0] == 1)
