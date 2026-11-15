# pyglotaran

pyglotaran is a Python library for global and target analysis of time-resolved spectroscopy data. It fits one model to one or more datasets at once. A _scheme_ describes the model, the _parameters_ hold the starting values, and the _result_ holds the optimized parameters, the fitted data, the residuals and the contribution of each element of the model as [xarray](https://docs.xarray.dev/en/stable/) datasets.

pyglotaran is the computation library and has no graphical interface. Plotting is in the companion package [pyglotaran-extras](https://github.com/glotaran/pyglotaran-extras).

## Install

pyglotaran needs Python 3.13 or 3.14. With [uv](https://docs.astral.sh/uv/), create an environment in a new folder and install pyglotaran into it:

```shell
uv venv --python 3.14
uv pip install --python .venv pyglotaran
```

`--python .venv` installs into the new environment also when another one is active. [Installation](installation.md) has the details, including how to activate the environment.

## A fit

```python
from glotaran.io import load_dataset, load_parameters, load_scheme

scheme = load_scheme("scheme.yml")
parameters = load_parameters("parameters.yml")
result = scheme.optimize(parameters, {"my_data": load_dataset("data.ascii")})
result.save("my_result")
```

The [quickstart](quickstart.md) is a complete script that runs as it is.

## Where to start

| You want to | Start with |
| --- | --- |
| Run a first fit from a script | [Quickstart](quickstart.md) |
| Analyse data in a notebook, with plots | {doc}`Getting started <notebooks/getting_started/getting_started>` |
| Learn one task at a time: data, schemes, parameters, fits, results | [Guides](user_documentation/index.md) |
| Port an analysis written for pyglotaran 0.7 | [Migrating from 0.7](migrating_from_0_7.md) |
| Record, compare and export fits | [Projects](user_documentation/project.md) |
| Build an application that calls pyglotaran | [Public API and file formats](reference/public_api.md) |
| See analyses of measured data | [pyglotaran-examples](https://github.com/glotaran/pyglotaran-examples) |

## Features

- Models are written as YAML scheme files, with a JSON Schema for completion and validation in an editor.
- Parameters are estimated with variable projection or non-negative least squares.
- Several datasets can be fitted together with shared parameters.
- Model elements and file formats are plugins, so you can add your own.
- Data and results are xarray datasets, built on NumPy and SciPy.

## Help and citing

- Questions and suggestions: [GitHub issues](https://github.com/glotaran/pyglotaran/issues), [Discord](https://discord.gg/KfnEYRSTJx) or the [mailing list](https://groups.google.com/forum/#!forum/glotaran).
- To cite pyglotaran, use "pyglotaran: a lego-like Python framework for global and target analysis of time-resolved spectra" (DOI: [10.1007/s43630-023-00460-y](https://doi.org/10.1007/s43630-023-00460-y)).
- Before you commit to pyglotaran for a study, read the [usage notice](https://github.com/glotaran/pyglotaran/blob/main/NOTICE.md#usage-notice-for-scientists).
- pyglotaran is developed in the same lab as [Glotaran](https://github.com/glotaran/glotaran-legacy), the Java interface to the R package TIMP, and does not replace it.

```{toctree}
:hidden:
:maxdepth: 2

get_started
user_documentation/index
reference/index
dev_docs/index
authors
```
