# Projects: recording, recomputing and exporting fits

A project is an optional folder that keeps track of your analysis. Every fit you run through a project is recorded with the minimum information needed to recompute its results, and a result you want to keep or share is exported deliberately, with everything needed to load it elsewhere. Without a project, pyglotaran works as before: `scheme.optimize` records nothing.

```python
from glotaran.io import load_parameters, load_scheme
from glotaran.project import Project

project = Project.start()  # the working directory, usually the folder of the notebook
scheme = load_scheme("models/scheme.yml")
parameters = load_parameters("models/parameters.yml")

result = project.optimize(scheme, parameters, datasets={"ta": ta})
project.list_results()
project.export(result, name="paper_fig3")
```

## Starting a project

`Project.start(folder)` opens the project in `folder`, by default the working directory. A project folder contains:

| Path | Written by | Content |
| --- | --- | --- |
| `project.gta` | you | Description of the project and its data |
| `results/` | `project.optimize` | One record per fit |
| `exports/` | `project.export` | One folder per exported result |

If the folder has no `project.gta`, `Project.start` creates one with the recommended fields left empty. An existing `project.gta` is never changed, so re-running the cell opens the same project. Parent folders are not searched: a notebook in a subfolder of the project passes the project folder, `Project.start("..")`. The folders are resolved to absolute paths when the project starts, so changing the working directory later has no effect.

A folder with a `project.gta` written by pyglotaran 0.7 opens as a project as well. Its 0.7 results are ignored by the listing.

### Describing the project and its data

`project.gta` is a YAML file for research data management (RDM). Every field is optional; pyglotaran reads none of them and copies the file into every export, so an exported result carries the description of its project. The recommended fields follow the DataCite metadata schema where it has an equivalent:

```yaml
title: Carotenoid dynamics in LH2 complexes
description: >
  Global and target analysis of ...
data_description: >
  Transient absorption, 400-700 nm, pump 800 nm; samples ...; instrument ...
creators:
  - name: ...
    orcid: 0000-0000-0000-0000
    affiliation: ...
keywords: [transient absorption, target analysis]
license: CC-BY-4.0
related_identifiers:
  - doi: 10.xxxx/...        # publication, dataset, data management plan
funding: ...
```

## Recording fits

`project.optimize(scheme, parameters, datasets, name=None, **kwargs)` runs `scheme.optimize` with the same arguments and records the fit. `name` is an optional label for the record. With `verbose=True` (the default of `optimize`) the record folder is printed.

A record is a folder in `results/` named by the start time of the fit, for example `results/2026-10-04_14-28-05/`; the folder name is the record id. A fit that starts in the same second as another gets `_2`, `_3`, and so on. A record contains:

| File | Content |
| --- | --- |
| `record.yml` | Status, notebook or script, scheme file, name, versions of pyglotaran and its dependencies, optimizer settings, a summary of each dataset and a summary of the fit |
| `scheme.yml` | The scheme |
| `initial_parameters.csv` | The starting values |
| `optimized_parameters.csv` | The optimized values and standard errors |
| `cost_history.csv` | The cost of every function evaluation |
| `parameter_history.csv` | The parameter values of every function evaluation, only with `verbose=True` |

A record contains no data and no result arrays, so recording every attempt takes little disk space. Instead, `record.yml` holds a summary of each dataset: its shape, the range, mean and RMS of its coordinates, of `data` and of `weight`, and the path of the file it was loaded from. This path is only a reference: if the data were preprocessed after loading (`isel`, scaling, combining), the file differs from the data that were fitted.

The record is written when the fit starts, with status `running`, and completed when it ends with status `success`, `failed` or `interrupted`. A failed or interrupted fit records the last parameters it evaluated. A record left at `running` belongs to a fit that is still running or never returned to Python, for example after the kernel died. If the record cannot be written, for example because the disk is full, you get a warning and the fit is not affected.

`result.record` holds the id and the path of the record. For a result of `scheme.optimize` it is `None`.

Several notebooks can share one project. Each record names the notebook or script that produced it, as detected in VS Code, Jupyter and for scripts; elsewhere (for example nbclient, papermill or Colab) it is `unknown`.

### Two record collections in one project

Each `Project` instance writes its records to its own results folder, so you can keep, for example, two approaches apart in one notebook:

```python
with_guide = Project.start(results="results_with_guide")
no_guide = Project.start(results="results_no_guide")
```

Both share `project.gta` and `exports/`. A results folder elsewhere is listed through its own instance, `Project.start("D:/archive/lycopene").list_results()`.

## Listing and comparing fits

`project.list_results()` returns a table of the records, sorted by start time, with the notebook, name, scheme file, status, the shape and RMS of each dataset, the cost, the RMSE and the number of function evaluations. A change in the shape or RMS columns shows where the data were replaced, for example with better measurements. The table can be filtered by `source`, `scheme_source` and `name` (text contained in the field) and by `status`.

`project.compare_results(a, b)` compares two fits, each given by record id, record path, `result.record` or an in-memory result. It shows the parameters that differ (value, standard error, expression, bounds, non-negative, vary), the fit statistics including the RMSE of each dataset, the difference between the schemes and what changed in the data. Changes in weights, penalties, scales and constraints show in the scheme difference or the data differences.

## Recomputing a recorded fit

Records contain no data, so to get back the full result of an earlier fit you supply its input data again:

```python
result = project.recompute("2026-10-04_14-31-40", datasets={"ta": ta})
```

`recompute` evaluates the recorded scheme once at the recorded optimized parameters. Before that, it compares the summary of each dataset with the recorded one and raises an error that names what changed, for example `ta: time max 10 -> 8 (-20.00%)`. Differences up to 1e-6 times the RMS of an array count as equal. Pass `allow_data_mismatch=True` to recompute anyway; the differences are listed in the result. After the evaluation, the cost and the RMSE of each dataset are compared with the recorded values, and a relative difference above 1e-6 gives a warning.

The recomputed result has `result.recomputation`, which holds the original fit (record id, summary, standard errors, cost history) and the reconstruction (time, versions, data differences, the comparison with the recorded values). It is saved and exported with the result. The Jacobian and the covariance matrix of the original fit are not recomputed. `recompute` also accepts an export folder.

An old attempt can only be recomputed if you can produce its input data again. Two habits make this easy:

- Keep the preprocessing (loading, selecting, scaling, combining) in a function, notebook or script that produces the input data without fitting. Re-running a whole notebook re-runs all its fits.
- Export the results of attempts that matter; an export contains the input data.

## Exporting a result

```python
project.export(result)                                # exports/last_result
project.export(result, name="paper_fig3")             # exports/paper_fig3
project.export(result, name="paper_fig3", overwrite=True)
```

`project.export` writes the result as it is in memory to a folder in `exports/`; an absolute `name` is used as given. The export is self-contained and loads with `load_result` on any computer:

- the input data as passed to `optimize`, including `weight`;
- the scheme and the initial and optimized parameters;
- the result arrays (fitted data, residuals, element and activation results, fit decomposition);
- a copy of `project.gta`;
- `export.yml` with the record id, the file names of the notebook and the scheme, the name, the versions, the optimizer settings and the data and fit summaries.

If an export with the same name exists, for example when you re-run the notebook, `export` warns that it may be stale and writes nothing. `overwrite=True` replaces it, but only a folder that contains an export.

`saving_options` takes the `SavingOptions` of `Result.save` to leave out result arrays or to choose file formats. The input data are always written. With `include_source_files=True`, the files the datasets were loaded from are copied to `source_files/<dataset label>/` as a reference; a file that cannot be found gives a warning and the export still succeeds. If you changed `result.optimized_parameters` after the fit, the export contains the changed values and lists them in `export.yml`.

To export an older attempt, recompute it first and export the recomputed result. Its `export.yml` holds the fit summary of the original fit and its record id as `recomputed_from`.
