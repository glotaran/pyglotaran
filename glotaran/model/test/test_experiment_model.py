from glotaran.model.clp_constraint import OnlyConstraint
from glotaran.model.clp_constraint import ZeroConstraint
from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.model.test.test_model import MockDataModel
from glotaran.model.test.test_model import MockModelWithDataModel
from glotaran.model.test.test_model import MockModelWithItem


def test_experiment_model_from_dict():
    library = {
        "m1": MockModelWithDataModel(label="m1", type="mock-w-datamodel"),
        "m2": MockModelWithItem(label="m2", type="mock-w-item"),
    }

    experiment_model = ExperimentModel.from_dict(
        library,
        {
            "datasets": {
                "d1": {"models": ["m1"]},
                "d2": {
                    "models": ["m2"],
                    "weights": [{"value": 1}, {"value": 2, "global_interval": (5, 6)}],
                },
                "d3": {"models": ["m2"], "global_models": ["m1"]},
            },
            "clp_penalties": [
                {
                    "type": "equal_area",
                    "source": "s",
                    "source_interval": (1, 2),
                    "target": "t",
                    "target_interval": [(1, 2)],
                    "parameter": "p",
                    "weight": 1,
                }
            ],
            "clp_constraints": [
                {
                    "type": "only",
                    "target": "t",
                    "interval": [(1, 2)],
                },
                {
                    "type": "zero",
                    "target": "t",
                    "interval": (1, 2),
                },
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

    only = experiment_model.clp_constraints[0]
    assert isinstance(only, OnlyConstraint)

    zero = experiment_model.clp_constraints[1]
    assert isinstance(zero, ZeroConstraint)
