"""Tests for ``glotaran.utils.pydantic_serde``."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PlainSerializer
from pydantic import PlainValidator
from pydantic import SerializationInfo
from pydantic import ValidationInfo
from pydantic import field_serializer
from pydantic import field_validator

from glotaran.io import load_parameters
from glotaran.parameter.parameters import Parameters
from glotaran.utils.pydantic_serde import deserialize_from_csv
from glotaran.utils.pydantic_serde import deserialize_parameters
from glotaran.utils.pydantic_serde import save_folder_from_info
from glotaran.utils.pydantic_serde import serialization_info_to_kwargs
from glotaran.utils.pydantic_serde import serialize_parameters
from glotaran.utils.pydantic_serde import serialize_to_csv

if TYPE_CHECKING:
    from collections.abc import Callable

    from glotaran.typing.types import Self
    from glotaran.typing.types import StrOrPath


def save_folder_from_info_wrapper(
    value: Any, info: SerializationInfo | ValidationInfo
) -> Path | None:
    """Wrapper around save_folder_from_info for use in pydantic serializers and validators."""
    return save_folder_from_info(info)


class ContextTestModel(BaseModel):
    """Dummy model to test context serialization and validation."""

    save_folder: Annotated[
        Path | None,
        PlainSerializer(save_folder_from_info_wrapper),
        PlainValidator(save_folder_from_info_wrapper),
    ]


class ToFromCsv:
    """Dummy class to test CSV serialization and deserialization."""

    data: pd.DataFrame

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize with a pandas DataFrame."""
        self.data = data

    def to_csv(self, path: StrOrPath, delimiter: str = ",") -> None:
        """Dummy method to simulate saving to CSV."""
        self.data.to_csv(path, index=False, sep=delimiter)

    @classmethod
    def from_csv(cls, path: StrOrPath) -> Self:
        """Dummy method to simulate loading from CSV."""
        return cls(pd.read_csv(path))


class ToFromCsvModel(BaseModel):
    """Dummy Pydantic model to test CSV serialization and deserialization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: ToFromCsv

    @field_serializer("data", when_used="json")
    def serialize_data(self, value: Any, info: SerializationInfo) -> Any:
        """Serialize the data field."""
        return serialize_to_csv(value, info)

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Any, info: ValidationInfo) -> Any:
        """Validate the data field."""
        return deserialize_from_csv(cls.model_fields[info.field_name].annotation, value, info)


class ModelWithParameters(BaseModel):
    """Dummy Pydantic model to test parameter serialization and deserialization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    params_1: Parameters
    params_2: Parameters

    @field_serializer("params_1", "params_2", when_used="json")
    def serialize_data(self, value: Parameters, info: SerializationInfo) -> Any:
        """Serialize the data field."""
        return serialize_parameters(value, info)

    @field_validator("params_1", "params_2", mode="before")
    @classmethod
    def validate_data(cls, value: Any, info: ValidationInfo) -> Any:
        """Validate the data field."""
        return deserialize_parameters(value, info)


@pytest.fixture
def parameters_model_data():
    """Roundtrip serialization and deserialization of parameters."""
    params_1 = Parameters.from_dict({"a": [3, 4, 5], "b": [7, 8]})
    params_2 = Parameters.from_dict({"c": [1, 2], "d": [4, 5]})
    model_instance = ModelWithParameters.model_validate(
        {"params_1": params_1, "params_2": params_2}
    )
    return model_instance, params_1, params_2


def test_serialization_info_to_kwargs():
    """Test serialization_info_to_kwargs function."""

    @dataclass
    class MockSerializationInfo:
        """Mock class that is compliant with the ``SerializationInfo`` Protocol."""

        mode: str
        context: dict[str, Any]
        include: Any = None
        exclude: Any = None
        exclude_defaults: bool = False
        exclude_none: bool = False
        exclude_unset: bool = False
        by_alias: bool = False
        round_trip: Callable[[], bool] = lambda: False
        serialize_as_any: bool = False

    info = MockSerializationInfo(
        mode="json",
        context={"save_folder": Path("test_folder"), "extra_option": 42},
    )

    assert serialization_info_to_kwargs(info) == asdict(info)

    filtered_kwargs = asdict(info)
    filtered_kwargs.pop("context", None)
    filtered_kwargs.pop("mode", None)
    assert serialization_info_to_kwargs(info, exclude={"context", "mode"}) == filtered_kwargs


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ({"save_folder": "test_folder"}, Path("test_folder")),
        ({}, None),
        (None, None),
    ],
)
def test_save_folder_from_info(
    tmp_path: Path, context: dict[str, str | Path], expected: Path | None
):
    """Check save_folder_from_info returns the expected values on serialization and validation."""
    test_instance = ContextTestModel.model_construct(save_folder=tmp_path)
    assert test_instance.save_folder == tmp_path
    assert test_instance.model_dump(context=context) == {"save_folder": expected}
    assert (
        ContextTestModel.model_validate({"save_folder": tmp_path}, context=context).save_folder
        == expected
    )


def test_serialize_to_csv(tmp_path: Path):
    """Test serialization to CSV using a Pydantic model_dump."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    to_from_csv_instance = ToFromCsv(data=data)
    model_instance = ToFromCsvModel(data=to_from_csv_instance)

    csv_path = tmp_path / "sub_dir" / "data.csv"
    serialized_value = model_instance.model_dump(
        context={"save_folder": tmp_path / "sub_dir"}, mode="json"
    )
    assert serialized_value["data"] == "data.csv"

    assert csv_path.exists()
    loaded_df = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(loaded_df, data)


def test_serialize_to_csv_missing_context_error():
    """Test serialization to CSV raises an error if save_folder is not in context."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    to_from_csv_instance = ToFromCsv(data=data)
    model_instance = ToFromCsvModel(data=to_from_csv_instance)

    with pytest.raises(ValueError) as exc_info:
        model_instance.model_dump(context={}, mode="json")
    assert "SerializationInfo context is missing 'save_folder'" in str(exc_info.value)


def test_deserialize_from_csv(tmp_path: Path):
    """Test deserialization from CSV using a Pydantic model_validate."""
    data = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    csv_path = tmp_path / "test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)

    model_instance = ToFromCsvModel.model_validate(
        {"data": "test.csv"}, context={"save_folder": tmp_path}
    )
    assert isinstance(model_instance.data, ToFromCsv)
    pd.testing.assert_frame_equal(model_instance.data.data, data)


def test_deserialize_from_csv_missing_context_error(tmp_path: Path):
    """Test deserialization from CSV raises an error if save_folder is not in context."""
    csv_path = tmp_path / "test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    data.to_csv(csv_path, index=False)

    with pytest.raises(ValueError) as exc_info:
        ToFromCsvModel.model_validate({"data": "test.csv"}, context={})
    assert "ValidationInfo context is missing 'save_folder'" in str(exc_info.value)


def test_parameters_serde_roundtrip_default(
    tmp_path: Path, parameters_model_data: tuple[ModelWithParameters, Parameters, Parameters]
):
    """Roundtrip serialization and deserialization of parameters."""
    model_instance, params_1, params_2 = parameters_model_data

    save_folder = tmp_path / "sub_dir"
    serialized_value = model_instance.model_dump(context={"save_folder": save_folder}, mode="json")
    assert serialized_value["params_1"] == "params_1.csv"
    assert serialized_value["params_2"] == "params_2.csv"

    assert (save_folder / "params_1.csv").is_file()
    assert (save_folder / "params_2.csv").is_file()
    assert load_parameters(save_folder / "params_1.csv") == params_1
    assert load_parameters(save_folder / "params_2.csv") == params_2

    loaded_model = ModelWithParameters.model_validate(
        serialized_value, context={"save_folder": save_folder}
    )
    assert loaded_model.params_1 == params_1
    assert loaded_model.params_2 == params_2


def test_parameters_serde_roundtrip_saving_options(
    tmp_path: Path, parameters_model_data: tuple[ModelWithParameters, Parameters, Parameters]
):
    """Roundtrip serde of ModelWithParameters parameters with custom extension name and plugin."""
    model_instance, params_1, params_2 = parameters_model_data

    save_folder = tmp_path / "sub_dir"
    context = {
        "save_folder": save_folder,
        "saving_options": {
            "parameter_format": "foo",
            "parameters_plugin": "glotaran.builtin.io.pandas.tsv.TsvProjectIo_tsv",
        },
    }

    serialized_value = model_instance.model_dump(context=context, mode="json")
    assert serialized_value["params_1"] == "params_1.foo"
    assert serialized_value["params_2"] == "params_2.foo"

    assert (save_folder / "params_1.foo").is_file()
    assert (save_folder / "params_2.foo").is_file()
    assert load_parameters(save_folder / "params_1.foo", format_name="tsv") == params_1
    assert load_parameters(save_folder / "params_2.foo", format_name="tsv") == params_2

    loaded_model = ModelWithParameters.model_validate(serialized_value, context=context)
    assert loaded_model.params_1 == params_1
    assert loaded_model.params_2 == params_2


def test_serialize_parameters_missing_context_error(
    parameters_model_data: tuple[ModelWithParameters, Parameters, Parameters],
):
    """Test serialization of parameters."""
    model_instance, _, _ = parameters_model_data

    with pytest.raises(ValueError) as exc_info:
        model_instance.model_dump(context={}, mode="json")
    assert "SerializationInfo context is missing 'save_folder'" in str(exc_info.value)


def test_deserialize_parameters_missing_context_error(
    tmp_path: Path, parameters_model_data: tuple[ModelWithParameters, Parameters, Parameters]
):
    """Test deserialization from CSV raises an error if save_folder is not in context."""
    model_instance, _, _ = parameters_model_data
    serialized_value = model_instance.model_dump(context={"save_folder": tmp_path}, mode="json")

    with pytest.raises(ValueError) as exc_info:
        ModelWithParameters.model_validate(serialized_value, context={})
    assert "ValidationInfo context is missing 'save_folder'" in str(exc_info.value)
