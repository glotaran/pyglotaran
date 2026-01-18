from __future__ import annotations

from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from tests.model.test_data_model import MockDataModel
from tests.model.test_data_model import MockElementWithDataModel
from tests.model.test_data_model import MockElementWithItem


def test_experiment_model_from_dict():
    library = {
        "m1": MockElementWithDataModel(label="m1", type="mock-w-datamodel"),
        "m2": MockElementWithItem(label="m2", type="mock-w-item"),
    }

    experiment_model = ExperimentModel.from_dict(
        library,
        {
            "datasets": {
                "d1": {"elements": ["m1"]},
                "d2": {
                    "elements": ["m2"],
                    "weights": [{"value": 1}, {"value": 2, "global_interval": (5, 6)}],
                },
                "d3": {"elements": ["m2"], "global_elements": ["m1"]},
            },
            "clp_penalties": [
                {
                    "type": "equal_area",
                    "source": "s",
                    "source_intervals": (1, 2),
                    "target": "t",
                    "target_intervals": [(1, 2)],
                    "parameter": "p",
                    "weight": 1,
                }
            ],
            "clp_relations": [
                {
                    "source": "s",
                    "target": "t",
                    "interval": [(1, 2)],
                    "parameter": "p",
                },
            ],
        },
    )

    d1 = experiment_model.datasets["d1"]
    assert isinstance(d1, MockDataModel)

    d2 = experiment_model.datasets["d2"]
    assert isinstance(d2, DataModel)
    assert not isinstance(d2, MockDataModel)

    d3 = experiment_model.datasets["d3"]
    assert isinstance(d3, MockDataModel)
