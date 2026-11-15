"""Write the scheme reference page from the JSON Schema of the installed pyglotaran.

The page lists every field of a scheme file with its type, default and description. It is
generated when the docs are built, so that it describes the elements and activations of the
installed version, including those of plugins.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING
from typing import Any

import glotaran
from glotaran.utils.json_schema import create_model_scheme_json_schema

if TYPE_CHECKING:
    from pathlib import Path

# Definitions that are parts of other definitions and get no section of their own.
SKIPPED = {"Element", "DataModel", "Parameter", "ParameterType", "LibraryType", "ModelLibrary"}

# Sections of the page: title, introduction and a function that selects the definitions.
FIXED_TITLES = {
    "ExperimentModel": "Experiment",
    "GlotaranDataModel": "Dataset",
    "ClpRelation": "Relation",
    "Weight": "Weight",
    "Oscillation": "Oscillation of a damped-oscillation element",
    "PFIDOscillation": "Oscillation of a pfid element",
}

# Text for definitions that have no description of their own.
NOTES = {
    "Scheme": "The top level of a scheme file.",
    "GlotaranDataModel": (
        "The model of one dataset. Some fields belong to elements: "
        "a dataset accepts them only when one of its elements uses them."
    ),
}

INTRODUCTION = """\
# Scheme reference

This page lists every field of a scheme file. It is generated from pyglotaran {version}, so it
matches the [JSON Schema](public_api.md#json-schema-of-the-scheme) of that version.
[Write a scheme](../user_documentation/scheme.md) explains how the parts work together.

In the tables:

- **parameter** is the label of a parameter in the parameter file, or a number for a fixed value.
- **required** in the column Default means the field has to be given.
- A **mapping** is written in YAML as `label: value` lines.

A scheme file has this structure:

```yaml
library:
  <element label>:
    type: <element type>
    # fields of the element

experiments:
  <experiment label>:
    # fields of the experiment
    datasets:
      <dataset label>:
        elements: [<element label>]
        # fields of the dataset
```
"""


def _anchor(title: str) -> str:
    """Return the label of the section with this title, which links refer to."""
    return "scheme-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _type_value(definition: dict[str, Any]) -> str | None:
    """Return the value of the ``type`` field of a typed definition."""
    return definition.get("properties", {}).get("type", {}).get("const")


def _titles(definitions: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Give every definition with fields the title of its section."""
    element_names = {
        spec["$ref"].rsplit("/", 1)[-1] for spec in definitions["ElementType"]["anyOf"]
    }
    titles = {}
    for name, definition in definitions.items():
        if name in SKIPPED or "properties" not in definition:
            continue
        type_value = _type_value(definition)
        if name in FIXED_TITLES:
            titles[name] = FIXED_TITLES[name]
        elif name in element_names:
            titles[name] = f"Element: {type_value}"
        elif name.endswith("Activation"):
            titles[name] = f"Activation: {type_value}"
        elif name.startswith("SpectralShape"):
            titles[name] = f"Shape: {type_value}"
        elif name.endswith("Constraint"):
            titles[name] = f"Constraint: {type_value}"
        elif name.endswith("Penalty"):
            titles[name] = f"Penalty: {type_value}"
        else:
            titles[name] = name
    return titles


REFERENCES = {
    "ParameterType": "parameter",
    "Parameter": "parameter label",
    "Element": "[element](#elements)",
    "ElementType": "[element](#elements)",
    "ModelLibrary": "mapping of label to [element](#elements)",
    "LibraryType": "mapping of label to [element](#elements)",
}


def _reference_text(name: str, titles: dict[str, str]) -> str:
    """Describe a reference to another definition, with a link to its section."""
    if name in REFERENCES:
        return REFERENCES[name]
    if name in titles:
        return f"{{ref}}`{titles[name]} <{_anchor(titles[name])}>`"
    return name


def _type_text(spec: dict[str, Any], titles: dict[str, str]) -> str:  # noqa: PLR0911
    """Describe the type of a field in words."""
    if "$ref" in spec:
        return _reference_text(spec["$ref"].rsplit("/", 1)[-1], titles)
    for key in ("anyOf", "oneOf"):
        if key in spec:
            parts = [_type_text(part, titles) for part in spec[key]]
            unique = [part for i, part in enumerate(parts) if part and part not in parts[:i]]
            return " or ".join(unique)
    if "const" in spec:
        return f"`{spec['const']}`"
    if "enum" in spec:
        return " or ".join(f"`{value}`" for value in spec["enum"])
    kind = spec.get("type")
    if kind == "null":
        return ""
    if kind == "array":
        if "prefixItems" in spec:
            return "[" + ", ".join(_type_text(item, titles) for item in spec["prefixItems"]) + "]"
        return f"list of {_type_text(spec.get('items', {}), titles) or 'values'}"
    if kind == "object" and isinstance(spec.get("additionalProperties"), dict):
        return f"mapping of label to {_type_text(spec['additionalProperties'], titles)}"
    return {"string": "text", "boolean": "`true` or `false`"}.get(kind, kind or "")


def _cell(text: str) -> str:
    """Make text safe for a cell of a Markdown table."""
    return " ".join(text.split()).replace("|", "\\|")


def _section(
    level: str, name: str, definition: dict[str, Any], titles: dict[str, str]
) -> list[str]:
    """Write the heading, description and field table of one definition."""
    lines = [f"({_anchor(titles[name])})=", "", f"{level} {titles[name]}", ""]
    description = definition.get("description", NOTES.get(name))
    if description:
        lines += [" ".join(description.split()), ""]
    required = set(definition.get("required", []))
    lines += ["| Field | Type | Default | Description |", "| --- | --- | --- | --- |"]
    for field, spec in definition["properties"].items():
        if field in required:
            default = "required"
        elif "default" not in spec and spec.get("type") in ("array", "object"):
            default = "empty"
        elif spec.get("default") is None:
            default = "none"
        else:
            default = f"`{json.dumps(spec['default'])}`"
        description = spec.get("description", "")
        if field == "type" and not description:
            description = "Selects this kind of item."
        lines.append(
            f"| `{field}` | {_cell(_type_text(spec, titles))} | {default} | {_cell(description)} |"
        )
    return [*lines, ""]


def write_scheme_reference(path: Path) -> None:
    """Write the scheme reference page as Markdown.

    Parameters
    ----------
    path : Path
        The file to write.
    """
    schema = create_model_scheme_json_schema()
    definitions = schema["$defs"]
    titles = {"Scheme": "Scheme", **_titles(definitions)}
    definitions = {"Scheme": schema, **definitions}

    def named(prefix: str) -> list[str]:
        return [name for name, title in titles.items() if title.startswith(prefix)]

    groups = {
        "Top level": ["Scheme", "ExperimentModel", "GlotaranDataModel"],
        "Elements": named("Element: "),
        "Activations": named("Activation: "),
        "Spectral shapes": named("Shape: "),
        "Constraints, relations and penalties": [
            *named("Constraint: "),
            "ClpRelation",
            *named("Penalty: "),
        ],
        "Weights and oscillations": ["Weight", "Oscillation", "PFIDOscillation"],
    }
    listed = {name for names in groups.values() for name in names}
    groups["Other definitions"] = [name for name in titles if name not in listed]

    lines = [INTRODUCTION.format(version=glotaran.__version__)]
    for group, names in groups.items():
        names = [name for name in names if name in titles]  # noqa: PLW2901
        if not names:
            continue
        lines += [f"## {group}", ""]
        for name in names:
            lines += _section("###", name, definitions[name], titles)
    path.write_text("\n".join(lines), encoding="utf-8")
