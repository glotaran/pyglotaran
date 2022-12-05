from glotaran.model.library import Library
from glotaran.model.test.test_item_new import MockItem
from glotaran.model.test.test_item_new import MockLibraryItem
from glotaran.model.test.test_item_new import MockLibraryItemNested
from glotaran.model.test.test_item_new import MockTypedItem
from glotaran.model.test.test_item_new import MockTypedItemConcrete1
from glotaran.model.test.test_item_new import MockTypedItemConcrete2


def test_construct_library():
    library_cls = Library.create([MockLibraryItem, MockTypedItem])
    assert hasattr(library_cls(), MockLibraryItem.get_library_name())
    assert isinstance(getattr(library_cls(), MockLibraryItem.get_library_name()), dict)


def test_initialize_library_from_dict():
    library_cls = Library.create([MockLibraryItem, MockTypedItem])
    library = library_cls.from_dict(
        {
            MockLibraryItem.get_library_name(): {"test_item": {"test_attr": "test_val"}},
            MockTypedItem.get_library_name(): {
                "test_item_typed1": {"type": "concrete_type1", "vint": 2},
                "test_item_typed2": {"type": "concrete_type2", "vstring": "teststr"},
            },
        }
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


def test_resolve_item():
    item = MockItem(
        cscalar=2,
        clist=[],
        cdict={},
        iscalar="test_lib_item",
        ilist=["test_lib_item"],
        idict={"foo": "test_lib_item"},
        pscalar="p",
        plist=[],
        pdict={},
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

    resolved = library.resolve_item(item)
    assert item is not resolved
    assert item.iscalar == "test_lib_item"

    assert isinstance(resolved.iscalar, MockLibraryItemNested)
    assert isinstance(resolved.iscalar.test_reference, MockLibraryItem)
    assert isinstance(resolved.ilist[0], MockLibraryItemNested)
    assert isinstance(resolved.ilist[0].test_reference, MockLibraryItem)
    assert isinstance(resolved.idict["foo"], MockLibraryItemNested)
    assert isinstance(resolved.idict["foo"].test_reference, MockLibraryItem)
