from typing import Literal

from glotaran.model.data_model import DataModel
from glotaran.model.library import Library
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.test.test_megacomplex import MockDataModel
from glotaran.model.test.test_megacomplex import MockMegacomplexWithDataModel
from glotaran.model.test.test_megacomplex import MockMegacomplexWithItem


class MockMegacomplexNonUniqueExclusive(Megacomplex):
    type: Literal["test_megacomplex_not_exclusive_unique"]

    def calculate_matrix():
        pass


class MockMegacomplexExclusive(Megacomplex):
    type: Literal["test_megacomplex_exclusive"]
    is_exclusive = True

    def calculate_matrix():
        pass


class MockMegacomplexUnique(Megacomplex):
    type: Literal["test_megacomplex_unique"]
    is_unique = True

    def calculate_matrix():
        pass


class MockMegacomplexDim1(Megacomplex):
    type: Literal["test_megacomplex_dim1"]
    dimension: str = "dim1"

    def calculate_matrix():
        pass


class MockMegacomplexDim2(Megacomplex):
    type: Literal["test_megacomplex_dim2"]
    dimension: str = "dim2"

    def calculate_matrix():
        pass


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


def test_get_data_model_issues():
    library = Library.from_dict(
        {
            "megacomplex": {
                "m": {"type": "test_megacomplex_not_exclusive_unique"},
                "m_exclusive": {"type": "test_megacomplex_exclusive"},
                "m_unique": {"type": "test_megacomplex_unique"},
            },
        },
        [
            MockMegacomplexNonUniqueExclusive,
            MockMegacomplexExclusive,
            MockMegacomplexUnique,
        ],
    )
    ok = DataModel.from_dict(library, {"megacomplex": ["m"]})
    exclusive = DataModel.from_dict(library, {"megacomplex": ["m", "m_exclusive"]})
    unique = DataModel.from_dict(library, {"megacomplex": ["m_unique", "m_unique"]})

    assert len(library.validate_item(ok)) == 0
    assert len(library.validate_item(exclusive)) == 1
    assert len(library.validate_item(unique)) == 2
