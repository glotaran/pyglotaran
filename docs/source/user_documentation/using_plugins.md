# Plugins

To be as flexible as possible `pyglotaran` uses a plugin system to handle new `Models`, `DataIo` and `ProjectIo`.
Those plugins can be defined by `pyglotaran` itself, the user or a 3rd party plugin package.

## Builtin plugins

### Models

- {class}`KineticSpectrumModel`
- {class}`KineticImageModel`

% TODO: Autogenerate support tables and use them for Io Plugin instead of bullet lists.

### Data Io

Plugins reading and writing data to and from {xarraydoc}`Dataset` or {xarraydoc}`DataArray`.

- {class}`AsciiDataIo`
- {class}`NetCDFDataIo`
- {class}`SdtDataIo`

### Project Io

Plugins reading and writing, {class}`Model`,:class:`Schema`,:class:`ParameterGroup` or {class}`Result`.

- {class}`YmlProjectIo`
- {class}`CsvProjectIo`
- {class}`FolderProjectIo`

## Reproducibility and plugins

With a plugin ecosystem there always is the possibility that multiple plugins try register under the same format/name.
This is why plugins are registered at least twice. Once under the name the developer intended and secondly
under their full name (full import path).
This allows to ensure that a specific plugin is used by manually specifying the plugin,
so if someone wants to run your analysis the results will be reproducible even if they have conflicting plugins installed.
You can gain all information about the installed plugins by calling the corresponding `*_plugin_table` function with both
options (`plugin_names` and `full_names`) set to true.
To pin a used plugin use the corresponding `set_*_plugin` function with the intended name (`format_name`/`model_name`)
and the full name (`full_plugin_name`) of the plugin to use.

If you wanted to ensure that the pyglotaran builtin plugin is used for `sdt` files you could add the following lines
to the beginning of your analysis code.

```python
from glotaran.io import set_data_plugin
set_data_plugin("sdt", "glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo_sdt")
```

### Models

The functions for model plugins are located in `glotaran.model` and called `model_plugin_table` and `set_model_plugin`.

### Data Io

The functions for data io plugins are located in `glotaran.io` and called `data_io_plugin_table` and `set_data_plugin`.

### Project Io

The functions for project io plugins are located in `glotaran.io` and called `project_io_plugin_table` and `set_project_plugin`.

## 3rd party plugins

Plugins not part of `pyglotaran` itself.

- Not yet, why not be the first? Tell us about your plugin and we will feature it here.
