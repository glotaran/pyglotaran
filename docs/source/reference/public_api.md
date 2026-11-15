# Public API and file formats

```{warning}
This page is a draft of the 1.0 contract. It describes pyglotaran 0.8.0 as released. Before 1.0 no stability guarantee applies, and the section "Open before 1.0" at the end of this page lists what can still change.
```

This page lists what an application built on pyglotaran can rely on: the Python names to import, the files pyglotaran reads and writes, and the rules for changing them. It is written for authors of notebooks and scripts, of graphical interfaces, and of tools in other languages that run pyglotaran as their computation engine.

The package is installed as `pyglotaran` and imported as `glotaran`.

## What is public

Public are:

- the names in the tables on this page, imported from the module given there, with their documented arguments and attributes;
- the file formats described under [Files](#files);
- the `type` names and field names of a scheme, as given by the [JSON Schema](#json-schema-of-the-scheme).

Everything else is internal and can change in any release. This includes:

- `glotaran.optimization`, except the result classes listed under [Results](#results);
- `glotaran.utils`, except `create_model_scheme_json_schema`;
- `glotaran.model`, except the classes in the tables on this page;
- `glotaran.plugin_system`; use the functions re-exported by `glotaran.io`;
- `glotaran.deprecation`, except `GlotaranApiDeprecationWarning`, and `glotaran.testing`;
- module paths below `glotaran.builtin`; a scheme selects elements and activations by their `type` name;
- names that start with an underscore, and the `serialize_*` and `validate_*` methods of the pydantic models.

## Stability rules

These rules are proposed for 1.0 and later.

- Versions follow [Semantic Versioning](https://semver.org). A change that breaks the public API or a file format needs a new major version.
- A public name that is deprecated keeps working until the next major version and emits `GlotaranApiDeprecationWarning`. The changelog names the replacement.
- A file written by a 1.x release loads in every later 1.x release.
- New optional fields, element types and result arrays can be added in a minor release. A file that uses them does not load in an older release: experiments, datasets, elements and results reject unknown fields.
- Numerical results can differ within floating-point tolerance between versions of pyglotaran, NumPy and SciPy. A [project record](../user_documentation/project.md) stores the versions that were used.

## Workflow

A fit needs a scheme, parameters and one dataset for each dataset label in the scheme.

```python
from glotaran.io import load_dataset, load_parameters, load_scheme, save_result

scheme = load_scheme("scheme.yml")
parameters = load_parameters("parameters.yml")
dataset = load_dataset("data.nc")

result = scheme.optimize(parameters, {"my_data": dataset})

result.optimization_info.success                # the optimizer returned without an error
result.optimization_info.termination_reason     # why it stopped
result.optimized_parameters.to_dataframe()      # values and standard errors
result.optimization_results["my_data"].residuals

save_result(result, "my_result/result.yml")
```

An application that holds the scheme in memory passes it as YAML or JSON text, `load_scheme(text, format_name="yml_str")`, or as a dictionary, `Scheme.from_dict(dictionary)`.

## Python API

### Reading and writing

| Name | Module | Purpose |
| --- | --- | --- |
| `load_scheme`, `save_scheme` | `glotaran.io` | Read and write a scheme file |
| `load_parameters`, `save_parameters` | `glotaran.io` | Read and write a parameter file |
| `load_dataset`, `save_dataset` | `glotaran.io` | Read and write a data file as `xarray.Dataset` |
| `load_result`, `save_result` | `glotaran.io` | Read and write a result folder |
| `SavingOptions`, `SAVING_OPTIONS_DEFAULT`, `SAVING_OPTIONS_MINIMAL` | `glotaran.io` | Which result arrays to write and in which file formats |

Every `load_*` and `save_*` function takes the file path and an optional `format_name`; without it the format follows from the file extension. Without `allow_overwrite=True`, the `save_*` functions raise `FileExistsError` if the given path is an existing file or a folder that is not empty.

For a result, this check covers only the path that is passed. `save_result(result, "my_result")` refuses a folder `my_result` that has any content. `save_result(result, "my_result/result.yml")` refuses only an existing `result.yml`, and replaces `scheme.yml`, the parameter files and the arrays of a result that is already in the folder. To keep earlier results, save each result to a new folder.

### Scheme and fit

| Name | Module | Purpose |
| --- | --- | --- |
| `Scheme` | `glotaran.project` | The model of an analysis: `library` and `experiments` |

`Scheme.optimize(parameters, datasets, **options)` runs a fit and returns a `Result`. The scheme, the parameters and the datasets are copied, so the call does not change them.

| Argument | Default | Meaning |
| --- | --- | --- |
| `parameters` | required | `Parameters` with the starting values |
| `datasets` | required | Mapping of dataset label to an `xarray.Dataset`, an `xarray.DataArray` or a file path |
| `optimization_method` | `"TrustRegionReflection"` | Also `"Dogbox"` and `"Levenberg-Marquardt"` |
| `maximum_number_function_evaluations` | `None` | Limit of function evaluations; `None` for the SciPy default |
| `ftol`, `gtol`, `xtol` | `1e-8` | Termination tolerances of `scipy.optimize.least_squares`; `None` disables one |
| `dry_run` | `False` | Evaluate the model once at the starting values instead of fitting. The result has all arrays and `optimization_info.success == False` |
| `verbose` | `True` | Print the iteration table to stdout and keep the parameter history |
| `raise_exception` | `False` | Raise an error of the optimizer. With `False` it becomes a warning and the result has `optimization_info.success == False`, as long as the model can still be evaluated at the last parameters; see [Errors and warnings](#errors-and-warnings) |

`Scheme.optimize` returns when the fit has ended. While the optimizer runs, `sys.stdout` of the process is replaced by an object that forwards to the original stream, also with `verbose=False`. An application that needs a timeout, a way to cancel a fit, or its own use of stdout during a fit runs the fit in a separate Python process.

### Parameters

| Name | Module | Purpose |
| --- | --- | --- |
| `Parameters` | `glotaran.parameter` | Collection of parameters, addressed by label |
| `Parameter` | `glotaran.parameter` | One parameter |
| `ParameterHistory` | `glotaran.parameter` | Parameter values of every function evaluation |

A `Parameter` has the fields `label`, `value`, `standard_error`, `expression`, `minimum`, `maximum`, `non_negative` and `vary`.

`Parameters` is created with `from_list`, `from_dict` or `from_dataframe`, or loaded from a file. `get(label)`, `has(label)`, `all()` and `labels` give access to the parameters, `to_dataframe()` returns a table with one row for each parameter, and `copy()` returns an independent copy.

### Results

| Name | Module | Purpose |
| --- | --- | --- |
| `Result` | `glotaran.project` | Everything a fit returns |
| `OptimizationInfo` | `glotaran.optimization` | Statistics of the fit |
| `OptimizerSettings` | `glotaran.optimization.info` | Method and tolerances the fit ran with |
| `OptimizationResult` | `glotaran.optimization.objective` | Result arrays of one dataset |
| `OptimizationHistory` | `glotaran.optimization.optimization_history` | Iteration table of the optimizer |

`Result` has these attributes:

| Attribute | Type | Content |
| --- | --- | --- |
| `scheme` | `Scheme` | Copy of the scheme the fit ran with |
| `initial_parameters` | `Parameters` | Starting values |
| `optimized_parameters` | `Parameters` | Fitted values with standard errors; fixed and expression parameters have none |
| `optimization_info` | `OptimizationInfo` | Statistics of the fit |
| `optimizer_settings` | `OptimizerSettings` | Method and tolerances |
| `optimization_results` | `dict[str, OptimizationResult]` | Result arrays by dataset label |
| `input_data` | `dict[str, xarray.Dataset or DataArray]` | Input data by dataset label |
| `record` | `RecordReference` or `None` | Record of a fit run through a `Project` |
| `recomputation` | `dict` or `None` | Set on a result of `Project.recompute` |

`Result.save(result_path)` writes the result like `save_result`.

`OptimizationInfo` has `success`, `termination_reason`, `number_of_function_evaluations`, `number_of_jacobian_evaluations`, `free_parameter_labels`, `cost`, `chi_square`, `reduced_chi_square`, `root_mean_square_error`, `degrees_of_freedom`, `number_of_data_points`, `number_of_parameters`, `number_of_clps`, `optimality`, `additional_penalty`, `covariance_matrix`, `jacobian`, `parameter_history`, `optimization_history` and `glotaran_version`. The rows and columns of `covariance_matrix` follow `free_parameter_labels`.

`success` is `True` when the optimizer returned a solution. It does not say whether the fit converged. `Scheme.optimize` ends in one of these ways:

| Outcome | `success` | `termination_reason` | Statistics |
| --- | --- | --- | --- |
| A termination condition is satisfied | `True` | Message of SciPy, for example `` `ftol` termination condition is satisfied. `` | All set |
| The limit of function evaluations is reached | `True` | `The maximum number of function evaluations is exceeded.` | All set |
| No parameter is free | `True` | `No free parameters to optimize.` | Set from one evaluation |
| `dry_run=True` | `False` | `Dry run.` | `cost`, `chi_square`, `reduced_chi_square`, `root_mean_square_error`, `degrees_of_freedom` and the other `number_of_*` fields are set; `number_of_jacobian_evaluations`, `optimality`, `jacobian` and `covariance_matrix` are `None` |
| Error of the optimizer | `False` | Message of the error | `cost` is the cost at the last evaluated parameters; the others are `None` |

`OptimizationResult` has:

| Attribute | Content |
| --- | --- |
| `input_data` | Data the fit used, with `weight` if the input had one |
| `residuals` | Data minus fit, not multiplied by `weight` |
| `fitted_data` | `input_data` minus `residuals` |
| `elements` | One `xarray.Dataset` for each element, by element label, with `amplitudes`, `concentrations` and element-specific arrays |
| `activations` | One `xarray.Dataset` for each activation, by activation label |
| `fit_decomposition` | `clp` and `matrix`: the conditionally linear parameters and the model matrix of the fit |
| `meta` | `model_dimension`, `global_dimension`, `root_mean_square_error`, `weighted_root_mean_square_error` and `scale` |

For data with a `weight`, `meta.root_mean_square_error` is computed from the residuals and `meta.weighted_root_mean_square_error` from the residuals multiplied by the weight. The `cost`, `chi_square` and `root_mean_square_error` of `OptimizationInfo` cover all datasets and use the weighted residuals.

### Projects

| Name | Module | Purpose |
| --- | --- | --- |
| `Project` | `glotaran.project` | Folder that records fits and exports results |
| `RecordReference` | `glotaran.project.record` | `id` and `path` of a record |
| `FitComparison` | `glotaran.project.compare` | Result of `Project.compare_results` |

`Project` has the methods `start`, `optimize`, `recompute`, `list_results`, `compare_results` and `export`, and the attributes `folder`, `results_folder`, `exports_folder` and `project_file`. [Projects](../user_documentation/project.md) describes them.

### Simulation

| Name | Module | Purpose |
| --- | --- | --- |
| `simulate` | `glotaran.simulation` | Compute a dataset from a data model, a library, parameters and coordinates |

### Building a scheme in code

| Name | Module | Purpose |
| --- | --- | --- |
| `ExperimentModel` | `glotaran.model.experiment_model` | Datasets that are fitted together |
| `DataModel` | `glotaran.model.data_model` | Model of one dataset |
| `ModelLibrary` | `glotaran.project.library` | Elements of a scheme, by label |

`Scheme.from_dict` takes a dictionary with the structure of a scheme file, for example parsed JSON.

### JSON Schema of the scheme

| Name | Module | Purpose |
| --- | --- | --- |
| `create_model_scheme_json_schema` | `glotaran.utils.json_schema` | JSON Schema of a scheme file |

`create_model_scheme_json_schema()` returns the schema as a dictionary, and with `output_path` it writes a file. The schema lists every element, activation and option of the installed pyglotaran, including those of plugins. With `parameters` it also lists the labels of those parameters, which an editor can offer as completions. An editor or a form generator can use the schema to validate and complete a scheme.

The schema is an aid for editors and form generators, and validates only part of what `load_scheme` checks. It checks field names and types, and not the references: the label of a parameter or an element that does not exist still validates. It also does not check the fields that an element requires of the datasets that use it: a dataset with a kinetic element and no `activations`, or a dataset with only a baseline element and `activations`, passes the schema and is rejected by `load_scheme`. A program that needs to know whether a scheme is valid loads it with `load_scheme`, then runs `Scheme.optimize` with `dry_run=True`, which also reports missing parameters and model issues, and catches the errors of both.

### Plugins

| Name | Module | Purpose |
| --- | --- | --- |
| `data_io_plugin_table`, `project_io_plugin_table` | `glotaran.io` | List the installed file formats |
| `set_data_plugin`, `set_project_plugin` | `glotaran.io` | Select one of two plugins that claim the same format |
| `register_data_io`, `DataIoInterface` | `glotaran.io` | Write a data format plugin |
| `register_project_io`, `ProjectIoInterface` | `glotaran.io` | Write a scheme, parameter or result format plugin |
| `Element` | `glotaran.model.element` | Base class of an element plugin |

Plugins are found through the entry point groups `glotaran.plugins.data_io`, `glotaran.plugins.project_io` and `glotaran.plugins.elements`.

### Errors and warnings

| Name | Module | Raised for |
| --- | --- | --- |
| `GlotaranUserError` | `glotaran.model.errors` | Input that cannot be used, such as a dataset label without data |
| `GlotaranModelError` | `glotaran.model.errors` | A model that is inconsistent |
| `GlotaranModelIssues` | `glotaran.model.errors` | A list of problems found when a fit starts; subclass of `GlotaranModelError` |
| `ParameterNotFoundError` | `glotaran.parameter.parameters` | A parameter label that is not in the parameters |
| `GlotaranApiDeprecationWarning` | `glotaran.deprecation` | Use of a deprecated name |

An unknown field in an experiment, a dataset, an element or an activation raises `pydantic.ValidationError`, as does an invalid parameter label. With `raise_exception=True`, an error of the optimizer is raised unchanged, for example a `ValueError` from SciPy for a starting value outside its bounds.

`raise_exception=False` covers only errors inside the optimizer. `Scheme.optimize` raises, whatever its value:

- before the fit, for input it cannot use: `GlotaranUserError` for a dataset label without data or a label used by two experiments, `ParameterNotFoundError` for a parameter label of the scheme that is not in the parameters, `GlotaranModelIssues` for other model problems such as a kinetic dataset without activations, and `UnsupportedMethodError` for an unknown `optimization_method`;
- after the optimizer stops, when the model cannot be evaluated at the last parameters. The fit then emits the warning and raises the error of the optimizer. An example is data with a NaN in an experiment with `residual_function: non_negative_least_squares`, which raises `ValueError`.

An application that must always get a result catches these exceptions around `Scheme.optimize`.

## Data

A dataset is an `xarray.Dataset` with a variable `data` of two dimensions. One is the model dimension, named by the `dimension` field of the elements of the dataset. Its default is `time` for the kinetic, damped-oscillation, coherent-artifact and pfid elements and `spectral` for the spectral element; the baseline and clp-guide elements have no default and take the dimension in `dimension`. All entries of `elements` of a dataset need the same dimension. The other is the global dimension, which `global_elements` describe. An optional variable `weight` with the same shape weights the residuals; it replaces the `weights` of the dataset in the scheme. An `xarray.DataArray` is taken as `data`.

Dataset labels name folders in a saved result. A label cannot be empty, end with `.` or a space, contain one of `<>:"/\|?*`, or be a Windows device name such as `CON`. Labels are unique across the experiments of a scheme and cannot differ only in case.

## Files

### Scheme file

A scheme file is YAML with the extension `.yml` or `.yaml` and two top-level keys.

```yaml
library:
  sequential:
    type: kinetic
    rates:
      (s2, s1): kinetic.1
      (s3, s2): kinetic.2
      (s3, s3): kinetic.3

experiments:
  my_experiment:
    datasets:
      my_data:
        elements: [sequential]
        activations:
          irf:
            type: gaussian
            center: irf.center
            width: irf.width
            compartments:
              s1: input.1
```

- `library` maps an element label to an element. `type` is one of `kinetic`, `spectral`, `baseline`, `coherent-artifact`, `damped-oscillation`, `pfid` and `clp-guide`, or the name of an element plugin.
- `experiments` maps an experiment label to an experiment. An experiment has `datasets` and the optional `residual_function` (`variable_projection` or `non_negative_least_squares`), `scale`, `clp_relations`, `clp_penalties`, `clp_link_method` and `clp_link_tolerance`.
- A dataset has `elements` and the optional `global_elements`, `element_scale`, `global_element_scale`, `weights` and `data` (a file path). It also accepts `residual_function`, which the fit does not use; set it on the experiment. Elements add fields: the kinetic, damped-oscillation and coherent-artifact elements add `activations`, the pfid element adds one `activation` of `type: multi-gaussian`.
- A value that is a parameter is the label of a parameter in the parameter file, or a number for a fixed value.
- A rate key `(to, from)` names the target and the source compartment.

The [JSON Schema](#json-schema-of-the-scheme) lists every field and type. `load_scheme` checks more, and the remaining checks run when a fit starts, also with `dry_run=True`: a missing parameter, and model issues such as an empty `activations: {}` of a kinetic dataset.

### Parameter files

A YAML parameter file is a list of parameters or a mapping of groups. A parameter is a value, or a list with an optional label, the value and an optional mapping of options.

```yaml
input:
  - ["1", 1, { vary: false }]

kinetic: [0.51, 0.31, 0.11]

irf:
  - [center, 0.31]
  - [width, 0.11, { min: 0, max: 1 }]
```

The label of a parameter is the labels of its groups and its own label joined by `.`, here `input.1`, `kinetic.1` to `kinetic.3`, `irf.center` and `irf.width`. A parameter without a label gets its position in the list, counted from 1. A label consists of ASCII letters, digits, `_` and `.`.

| Option | Meaning | Default |
| --- | --- | --- |
| `vary` | Optimize the parameter | `true` |
| `min`, `max` | Bounds | none |
| `non-negative` | Keep the value positive by optimizing its logarithm | `false` |
| `expr` | Expression that computes the value; `$label` refers to another parameter | none |

A table file (`.csv`, `.tsv`, `.xlsx`, `.ods`) has one row for each parameter with the columns `label` and `value`, and optionally `standard_error`, `expression`, `minimum`, `maximum`, `non_negative` and `vary`. Saved parameters use this layout.

| Format | Load | Save |
| --- | --- | --- |
| `yml`, `yaml`, `yml_str` | yes | no |
| `csv`, `tsv`, `xlsx`, `ods` | yes | yes |

### Data files

| Format | Load | Save | Content |
| --- | --- | --- | --- |
| `nc` | yes | yes | NetCDF, as written by xarray |
| `ascii` | yes | yes | Wavelength-explicit or time-explicit text file |
| `sdt` | yes | no | Becker & Hickl SDT file |

### Result folder

`save_result(result, "my_result/result.yml")` writes:

```text
my_result/
  result.yml                      index of the files below and the fit statistics
  scheme.yml
  initial_parameters.csv
  optimized_parameters.csv
  parameter_history.csv           starting values; with verbose=True one row for each function evaluation
  optimization_history.csv        iteration table of the optimizer; empty with verbose=False
  optimization_results/
    <dataset label>/
      input_data.nc               variable data, and weight if present
      residuals.nc                variable residual
      fitted_data.nc
      elements/<element label>.nc
      activations/<activation label>.nc
      fit_decomposition/clp.nc
      fit_decomposition/matrix.nc
```

`result.yml` holds the relative path of each file, the `meta` values of each dataset, `optimization_info` and `optimizer_settings`. `load_result` takes the folder or the path of `result.yml`.

`SavingOptions` selects the file formats (`data_format`, `parameters_format`, `scheme_format`) and, with `data_filter`, the arrays to leave out: `input_data`, `residuals`, `fitted_data`, `elements`, `activations` and `fit_decomposition`. `SAVING_OPTIONS_MINIMAL` leaves out all six.

Two entries of `data_filter` differ from the others:

- `fitted_data.nc` is not read by `load_result`. A loaded result computes `fitted_data` from `input_data` and `residuals`, so a result saved without `residuals` has no fitted data after loading.
- `input_data` is left out only for data that were loaded with `load_dataset`. `result.yml` then holds the path of the original file relative to the result folder, and the result loads only while that file stays in place. Data created in memory are always written.

The covariance matrix and the Jacobian are not saved; `optimized_parameters.csv` holds the standard errors.

### Project folder

A project folder holds `project.gta`, one record folder for each fit in `results/` and one folder for each export in `exports/`. `record.yml` and `export.yml` carry `schema_version: 1`. A record folder holds the scheme, the parameters and the histories of a fit, without data or result arrays; `Project.recompute` needs the input data again. An export is a result folder with `export.yml` and a copy of `project.gta` added, and includes the input data. [Projects](../user_documentation/project.md) describes the files.

## Open before 1.0

These points are decided before the contract is frozen.

1. **Import locations.** `OptimizationResult`, `OptimizerSettings`, `OptimizationHistory`, `ExperimentModel`, `DataModel`, `ModelLibrary`, `Element`, `RecordReference`, `FitComparison`, the error classes and `create_model_scheme_json_schema` are imported from implementation modules. Re-exporting them from `glotaran.project`, `glotaran.model`, `glotaran.optimization` and `glotaran.io` gives each public name one stable location.
2. **Extent of the public API.** The other names exported by `glotaran.io` (`prepare_time_trace_dataset`, `load_datasets`, `get_dataloader`, `get_datasaver`, `get_project_io_method`, the `show_*_help` functions), the remaining methods of `Parameters` and `Parameter`, and `glotaran.testing.simulated_data` are public or internal. Importing `glotaran.testing.simulated_data.sequential_spectral_decay` or `parallel_spectral_decay` runs a fit and prints its iteration table.
3. **Version of the result format.** `result.yml` has no format version; only `optimization_info.glotaran_version` identifies the writer.
4. **Unnamed arrays.** `fitted_data.nc`, `clp.nc` and `matrix.nc` store their array under xarray's placeholder name `__xarray_dataarray_variable__`.
5. **Missing values in parameter tables.** A parameter without expression or standard error is written as the text `None`.
6. **Covariance matrix.** A loaded result has no covariance matrix, so parameter correlations are lost on saving.
7. **Names.** Element and activation types use hyphens (`coherent-artifact`, `multi-gaussian`), the penalty type uses an underscore (`equal_area`), and the activation field `reciproke_global_axis` is misspelled.
8. **Errors.** The exception classes have no common base class, and a model issue has a message but no code or location in the scheme. `GlotaranDefinitionError` is defined and never raised. Malformed input raises built-in exceptions: `KeyError` for a dataset without the model dimension and for a dataset model that names an element missing from the library, `AttributeError` for a dataset without the variable `data`, `StopIteration` for data with one dimension, and `ValueError` for a `weight` of another shape.
9. **Input that is accepted without a message.** An unknown top-level key of a scheme file is ignored, and a missing `library` or `experiments` raises `KeyError`. An unknown option of a parameter is ignored, so `{ vari: false }` leaves the parameter free. A dataset passed to `optimize` under a label that the scheme does not use is ignored. `residual_function` of a dataset is accepted and not used. A dataset with `global_elements` ignores the `clp_constraints` of its elements and the `clp_relations` and `clp_penalties` of its experiment.
10. **Fit outcome.** `success` is `True` for a fit that stopped at the limit of function evaluations and `False` for a dry run that returns valid arrays. `OptimizationInfo` has no field for the convergence flag of SciPy; only `summary.converged` in `record.yml` holds it. `termination_reason` is the message text of SciPy, without a code that a program can test.
11. **Progress and cancelling.** A running fit reports progress only by printing SciPy's iteration table, from which `optimization_history.csv` is parsed, and can be stopped only with `KeyboardInterrupt`. Every fit replaces `sys.stdout` for the whole process, including its other threads.
12. **Command line.** pyglotaran 0.8 has no command-line interface; a program in another language runs a Python script.
13. **Parameters as YAML.** Parameters can be loaded from YAML but saved only as a table.
14. **Checking input before a fit.** No function checks a scheme, the parameters and the data together without fitting. `dry_run=True` evaluates the model once and stops at the first problem. The values of the data are not checked: with a NaN in the data the fit ends with `success == False` and the reason `Residuals are not finite in the initial point.`, or raises `ValueError` with `residual_function: non_negative_least_squares`.
15. **JSON Schema and loader.** The JSON Schema rejects an unknown top-level key and a missing `library`, which the loader ignores or reports as `KeyError`. It accepts references to parameters and elements that do not exist, which the loader or the fit rejects, and datasets that lack or have extra fields required by their elements, which the loader rejects.
16. **Saving options.** A result saved without `residuals` still writes `fitted_data.nc`, which `load_result` does not read. `input_data` in `data_filter` has no effect for data created in memory, and for data loaded from a file the saved result depends on a file outside the result folder.
