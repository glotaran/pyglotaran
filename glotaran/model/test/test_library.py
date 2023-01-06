import pytest

from glotaran.model.library import Library
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.test.test_item import MockItem
from glotaran.model.test.test_item import MockLibraryItem
from glotaran.model.test.test_item import MockLibraryItemNested
from glotaran.model.test.test_item import MockTypedItem
from glotaran.model.test.test_item import MockTypedItemConcrete1
from glotaran.model.test.test_item import MockTypedItemConcrete2
from glotaran.model.test.test_megacomplex import MockDataModel
from glotaran.model.test.test_megacomplex import MockMegacomplexWithDataModel
from glotaran.model.test.test_megacomplex import MockMegacomplexWithItem
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters


def test_construct_library():
    library_cls = Library.create([MockLibraryItem, MockTypedItem])
    assert hasattr(library_cls(), MockLibraryItem.get_library_name())
    assert isinstance(getattr(library_cls(), MockLibraryItem.get_library_name()), dict)


@pytest.mark.parametrize(
    "megacomplexes",
    ([MockMegacomplexWithDataModel, MockMegacomplexWithItem], None),
)
def test_construct_library_for_megacomplexes(megacomplexes):
    library_cls = Library.create_for_megacomplexes(megacomplexes)
    for item in [Megacomplex, MockTypedItem, MockLibraryItemNested, MockLibraryItem]:
        assert hasattr(library_cls(), item.get_library_name())


def test_initialize_library_from_dict():
    library = Library.from_dict(
        {
            MockLibraryItem.get_library_name(): {"test_item": {"test_attr": "test_val"}},
            MockTypedItem.get_library_name(): {
                "test_item_typed1": {"type": "concrete_type1", "vint": 2},
                "test_item_typed2": {"type": "concrete_type2", "vstring": "teststr"},
            },
        },
        megacomplexes=[MockMegacomplexWithDataModel, MockMegacomplexWithItem],
    )

    test_items = getattr(library, MockLibraryItem.get_library_name())
    assert "test_item" in test_items
    assert isinstance(test_items["test_item"], MockLibraryItem)
    assert test_items["test_item"].label == "test_item"
    assert test_items["test_item"].test_attr == "test_val"

    test_items_typed = getattr(library, MockTypedItem.get_library_name())
    assert "test_item_typed1" in test_items_typed
    assert isinstance(test_items_typed["test_item_typed1"], MockTypedItemConcrete1)
    assert test_items_typed["test_item_typed1"].label == "test_item_typed1"
    assert test_items_typed["test_item_typed1"].vint == 2

    assert "test_item_typed2" in test_items_typed
    assert isinstance(test_items_typed["test_item_typed2"], MockTypedItemConcrete2)
    assert test_items_typed["test_item_typed2"].label == "test_item_typed2"
    assert test_items_typed["test_item_typed2"].vstring == "teststr"


def test_get_data_model():
    library = Library.from_dict(
        {
            "megacomplex": {
                "m1": {"type": "mock-w-datamodel"},
                "m2": {"type": "mock-w-item"},
            },
        },
        megacomplexes=[MockMegacomplexWithDataModel, MockMegacomplexWithItem],
    )

    d1 = library.get_data_model_for_megacomplexes(["m1"])
    assert issubclass(d1, MockDataModel)

    d2 = library.get_data_model_for_megacomplexes(["m2"])
    assert not issubclass(d2, MockDataModel)

    d3 = library.get_data_model_for_megacomplexes(["m1", "m2"])
    assert issubclass(d3, MockDataModel)


def test_get_issues():
    library = Library.create([MockLibraryItem]).from_dict({})
    item = MockItem(
        cscalar=0,
        clist=[],
        cdict={},
        iscalar="foo",
        ilist=["foo", "bar"],
        idict={1: "foo", 2: "bar"},
        pscalar="foo",
        plist=["foo", "bar"],
        pdict={1: "foo", 2: "bar"},
    )

    issues = library.validate_item(item)
    assert len(issues) == 5

    issues = library.validate_item(item, parameters=Parameters({}))
    assert len(issues) == 10


def test_resolve_item():
    item = MockItem(
        cscalar=2,
        clist=[],
        cdict={},
        iscalar="test_lib_item",
        ilist=["test_lib_item"],
        idict={"foo": "test_lib_item"},
        pscalar="param1",
        plist=["param1", "param2"],
        pdict={"foo": "param2"},
    )

    library_cls = Library.create([MockLibraryItem, MockLibraryItemNested])
    library = library_cls.from_dict(
        {
            MockLibraryItem.get_library_name(): {
                "test_lib_item_nested": {"test_attr": "test_val"},
            },
            MockLibraryItemNested.get_library_name(): {
                "test_lib_item": {
                    "test_reference": "test_lib_item_nested",
                }
            },
        }
    )

    parameters = Parameters({})
    initial = Parameters.from_list(
        [
            ["param1", 1.0],
            ["param2", 2.0],
        ]
    )

    resolved = library.resolve_item(item, parameters, initial)
    assert item is not resolved
    assert item.iscalar == "test_lib_item"

    assert isinstance(resolved.iscalar, MockLibraryItemNested)
    assert isinstance(resolved.iscalar.test_reference, MockLibraryItem)
    assert isinstance(resolved.ilist[0], MockLibraryItemNested)
    assert isinstance(resolved.ilist[0].test_reference, MockLibraryItem)
    assert isinstance(resolved.idict["foo"], MockLibraryItemNested)
    assert isinstance(resolved.idict["foo"].test_reference, MockLibraryItem)

    assert parameters.has("param1")
    assert parameters.get("param1") is not initial.get("param1")
    assert parameters.has("param2")
    assert parameters.get("param2") is not initial.get("param2")

    assert isinstance(resolved.pscalar, Parameter)
    assert resolved.pscalar.value == 1.0

    assert isinstance(resolved.plist[0], Parameter)
    assert resolved.plist[0].value == 1.0
    assert isinstance(resolved.plist[1], Parameter)
    assert resolved.plist[1].value == 2.0

    assert isinstance(resolved.pdict["foo"], Parameter)
    assert resolved.pdict["foo"].value == 2.0

    parameters.get("param1").value = 10
    assert parameters.get("param1").value != initial.get("param1").value
    assert resolved.pscalar.value == 10
