from __future__ import annotations

from dataclasses import dataclass

from glotaran.project.dataclass_helpers import asdict
from glotaran.project.dataclass_helpers import exclude_from_dict_field
from glotaran.project.dataclass_helpers import file_loadable_field
from glotaran.project.dataclass_helpers import fromdict
from glotaran.project.dataclass_helpers import init_file_loadable_fields


@dataclass
class DummyFileLoadable:
    def __init__(self, val: str) -> None:
        self.source_path = "dummy_file"
        self.data = {"foo": val}

    @classmethod
    def loader(
        cls: type[DummyFileLoadable],
        file_path: str,
    ) -> DummyFileLoadable:
        instance = cls(f"{file_path}_loaded")
        instance.source_path = file_path
        return instance


def test_serialize_to_file_name_field():
    @dataclass
    class DummyDataclass:
        foo: DummyFileLoadable = file_loadable_field(DummyFileLoadable)
        foo2: DummyFileLoadable = file_loadable_field(DummyFileLoadable)
        bar: int = exclude_from_dict_field(default=42)
        baz: int = 84

        def __post_init__(self):
            init_file_loadable_fields(self)

    dummy_class = DummyDataclass(foo=DummyFileLoadable.loader("foo.file"), foo2="foo2.file")

    assert dummy_class.foo.data == {"foo": "foo.file_loaded"}
    assert dummy_class.foo.source_path == "foo.file"
    assert dummy_class.foo2.data == {"foo": "foo2.file_loaded"}
    assert dummy_class.foo2.source_path == "foo2.file"
    assert dummy_class.bar == 42

    dummy_class_dict = asdict(dummy_class)

    assert dummy_class_dict["foo"] == "foo.file"
    assert dummy_class_dict["foo2"] == "foo2.file"
    assert "bar" not in dummy_class_dict

    assert dummy_class_dict["baz"] == 84
    assert dummy_class_dict["baz"] == dummy_class.baz

    loaded = fromdict(DummyDataclass, dummy_class_dict)

    assert loaded == dummy_class
