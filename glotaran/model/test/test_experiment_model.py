from glotaran.model.clp_constraint import OnlyConstraint
from glotaran.model.clp_constraint import ZeroConstraint
from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.model.library import Library
from glotaran.model.test.test_megacomplex_new import MockDataModel
from glotaran.model.test.test_megacomplex_new import MockMegacomplexWithDataModel
from glotaran.model.test.test_megacomplex_new import MockMegacomplexWithItem


def test_experiment_model_from_dict():
    library = Library.from_dict(
        {
            "megacomplex": {
                "m1": {"type": "mock-w-datamodel"},
                "m2": {"type": "mock-w-item"},
            },
        },
        megacomplexes=[MockMegacomplexWithDataModel, MockMegacomplexWithItem],
    )

    experiment_model = ExperimentModel.from_dict(
        library,
        {
            "datasets": {
                "d1": {"megacomplex": ["m1"], "item": "foo"},
                "d2": {"megacomplex": ["m2"]},
                "d3": {"megacomplex": ["m2"], "global_megacomplex": ["m1"], "item": "foo"},
            },
            "clp_penalties": [
                {
                    "type": "equal_area",
                    "source": "s",
                    "source_intervals": [(1, 2)],
                    "target": "t",
                    "target_intervals": [(1, 2)],
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
            "weights": [
                {"datasets": ["d1", "d2"], "value": 1},
                {"datasets": ["d3"], "value": 2, "global_interval": (5, 6)},
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
