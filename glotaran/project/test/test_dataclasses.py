from dataclasses import dataclass

from glotaran.project.dataclasses import asdict
from glotaran.project.dataclasses import serialize_to_file_name_field


def test_serialize_to_file_name_field():
    @dataclass
    class DummyDataclass:
        foo: int = serialize_to_file_name_field("foo.file")
        bar: int = serialize_to_file_name_field("bar.file", default=42)
        baz: int = 84

    dummy_class = DummyDataclass(foo=21)

    dummy_class_dict = asdict(dummy_class)

    assert dummy_class_dict["foo"] == "foo.file"
    assert dummy_class_dict["foo"] != dummy_class.foo

    assert dummy_class_dict["bar"] == "bar.file"
    assert dummy_class_dict["bar"] != dummy_class.bar

    assert dummy_class_dict["baz"] == 84
    assert dummy_class_dict["baz"] == dummy_class.baz
