from glotaran.model.data_model import DataModel
from glotaran.model.library import Library
from glotaran.model.test.test_megacomplex_new import MockDataModel
from glotaran.model.test.test_megacomplex_new import MockMegacomplexWithDataModel
from glotaran.model.test.test_megacomplex_new import MockMegacomplexWithItem


def test_data_model_from_dict():
    library = Library.from_dict(
        {
            "megacomplex": {
                "m1": {"type": "mock-w-datamodel"},
                "m2": {"type": "mock-w-item"},
            },
        },
        megacomplexes=[MockMegacomplexWithDataModel, MockMegacomplexWithItem],
    )

    d1 = DataModel.from_dict(library, {"megacomplex": ["m1"], "item": "foo"})
    assert isinstance(d1, MockDataModel)

    d2 = DataModel.from_dict(library, {"megacomplex": ["m2"]})
    assert isinstance(d2, DataModel)
    assert not isinstance(d2, MockDataModel)

    d3 = DataModel.from_dict(
        library, {"megacomplex": ["m2"], "global_megacomplex": ["m1"], "item": "foo"}
    )
    assert isinstance(d3, MockDataModel)
