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
            }
        },
    )

    d1 = experiment_model.datasets["d1"]
    assert isinstance(d1, MockDataModel)

    d2 = experiment_model.datasets["d2"]
    assert isinstance(d2, DataModel)
    assert not isinstance(d2, MockDataModel)

    d3 = experiment_model.datasets["d3"]
    assert isinstance(d3, MockDataModel)
