from dataclasses import dataclass

from glotaran.project.dataclass_helpers import asdict
from glotaran.project.dataclass_helpers import exclude_from_dict_field
from glotaran.project.dataclass_helpers import file_representation_field
from glotaran.project.dataclass_helpers import fromdict


def dummy_loader(file: str) -> int:
    return {"foo.file": 21, "bar.file": 42}[file]


def test_serialize_to_file_name_field():
    @dataclass
    class DummyDataclass:
        foo: int = exclude_from_dict_field()
        foo_file: int = file_representation_field("foo", dummy_loader)
        bar: int = exclude_from_dict_field(default=42)
        bar_file: int = file_representation_field("bar", dummy_loader, default="bar.file")
        baz: int = 84

    dummy_class = DummyDataclass(foo=21, foo_file="foo.file")

    dummy_class_dict = asdict(dummy_class)

    assert "foo" not in dummy_class_dict
    assert dummy_class_dict["foo_file"] == "foo.file"

    assert "bar" not in dummy_class_dict
    assert dummy_class_dict["bar_file"] == "bar.file"

    assert dummy_class_dict["baz"] == 84
    assert dummy_class_dict["baz"] == dummy_class.baz

    loaded = fromdict(DummyDataclass, dummy_class_dict)

    assert loaded == dummy_class
