"""Utility functionality module for ``glotaran.builtin.io.yml.yml``"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

if TYPE_CHECKING:
    from typing import Any
    from typing import Mapping
    from typing import Sequence

    from ruamel.yaml.nodes import ScalarNode
    from ruamel.yaml.representer import BaseRepresenter


def write_dict(
    data: Mapping[str, Any] | Sequence[Any], file_name: str | Path | None = None, offset: int = 0
) -> str | None:
    yaml = YAML()
    yaml.representer.add_representer(type(None), _yaml_none_representer)
    yaml.indent(mapping=2, sequence=2, offset=offset)

    if file_name is not None:
        with open(file_name, "w") as f:
            yaml.dump(data, f)
    else:
        stream = StringIO()
        yaml.dump(data, stream)
        return stream.getvalue()


def load_dict(source: str | Path, is_file: bool) -> dict[str, Any]:
    yaml = YAML()
    yaml.representer.add_representer(type(None), _yaml_none_representer)
    if is_file:
        with open(source) as f:
            spec = yaml.load(f)
    else:
        spec = yaml.load(source)
    return spec


def _yaml_none_representer(representer: BaseRepresenter, data: Mapping[str, Any]) -> ScalarNode:
    """Yaml repr for ``None`` python values.

    Parameters
    ----------
    representer : BaseRepresenter
        Representer of the :class:`YAML` instance.
    data : Mapping[str, Any]
        Data to write to yaml.

    Returns
    -------
    ScalarNode
        Node representing the value.

    References
    ----------
    https://stackoverflow.com/a/44314840
    """
    return representer.represent_scalar("tag:yaml.org,2002:null", "null")
