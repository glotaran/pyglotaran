# Migrating from 0.7

pyglotaran 0.8 replaces the model file and the result layout of 0.7. Parameter files and data files load unchanged. To keep an analysis on 0.7, pin `pyglotaran<0.8` and `pyglotaran-extras<0.8`.

The fitting method is the same. The model on this page, fitted to the same data with 0.7.5 and with 0.8.0, gives the same parameters, root mean square error, spectra, concentrations and residuals.

## Overview

| 0.7 | 0.8 |
| --- | --- |
| `load_model("model.yml")` | `load_scheme("scheme.yml")` |
| `megacomplex` of type `decay`, `decay-sequential` or `decay-parallel`, with `k_matrix` | `library` element of type `kinetic` with `rates` |
| `irf` and `initial_concentration` of a dataset | `activations` of the dataset |
| `dataset` | `datasets` of an experiment |
| `dataset_groups` with linked amplitudes (`link_clp: true`, or the default for datasets that can be linked) | One experiment with those datasets |
| A dataset group with `link_clp: false` | One experiment for each dataset; the parameters stay shared. See [Fit several datasets](user_documentation/multiple_datasets.md) |
| `clp_link_tolerance`, `clp_link_method` on `Scheme` | Fields of the experiment |
| `clp_area_penalties` | `clp_penalties` of the experiment |
| `optimize(Scheme(model, parameters, data))` | `scheme.optimize(parameters, datasets)` |
| `result.data["my_data"]` | `result.optimization_results["my_data"]` |
| `result.root_mean_square_error` and the other statistics | `result.optimization_info.root_mean_square_error` |
| `save_result(result, "folder/result.yml")` | `result.save("folder")` |
| `glotaran.parameter.ParameterGroup` | `glotaran.parameter.Parameters` |
| `glotaran.examples` | `glotaran.testing.simulated_data` |

## Errors you see with 0.7 files and code

| Error in 0.8 | Cause |
| --- | --- |
| `KeyError: 'library'` from `load_scheme` | The file is a 0.7 model file. Port it as shown below |
| `KeyError: 'library'` from `load_result` | The result was saved by 0.7. pyglotaran 0.8 does not read 0.7 results; fit again or read the files with 0.7 |
| `AttributeError: module 'glotaran.io' has no attribute 'load_model'` | Use `load_scheme` |
| `ModuleNotFoundError: No module named 'glotaran.optimization.optimize'` | Use `scheme.optimize` |
| `AttributeError: 'Result' object has no attribute 'data'` | Code written for a 0.7 result, or pyglotaran-extras 0.7 with a 0.8 result. See [Results](#results) and [Plotting](#plotting) |

## Model file to scheme file

The model of the getting started guide in 0.7:

```yaml
initial_concentration:
  input:
    compartments: [s1, s2, s3]
    parameters: [input.1, input.0, input.0]

k_matrix:
  k1:
    matrix:
      (s2, s1): kinetic.1
      (s3, s2): kinetic.2
      (s3, s3): kinetic.3

megacomplex:
  m1:
    k_matrix: [k1]
    type: decay

irf:
  irf1:
    type: gaussian
    center: irf.center
    width: irf.width

dataset:
  my_data:
    initial_concentration: input
    megacomplex: [m1]
    irf: irf1
```

The same model as a 0.8 scheme:

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

1. The megacomplex and its `k_matrix` become one element in `library` with `type: kinetic`. The entries of `matrix` become `rates`, with the same keys.
2. The `irf` and the `initial_concentration` of the dataset become one activation of the dataset. `center` and `width` are those of the IRF, and `compartments` holds the input of each compartment. A compartment that is not listed gets no input, so the parameter `input.0` is no longer needed. A dataset without an IRF uses an activation of `type: instant`.
3. The dataset moves to `datasets` of an experiment, and `megacomplex: [m1]` becomes `elements: [sequential]`. An experiment holds the datasets that are fitted together.
4. The megacomplex types `spectral`, `baseline`, `coherent-artifact`, `damped-oscillation`, `clp-guide` and `pfid` exist as element types of the same name. The [JSON Schema](reference/public_api.md#json-schema-of-the-scheme) lists the fields of each element.

The 0.7 parameter file loads in 0.8 as it is. Parameters that the scheme does not use, here `input.0`, are ignored.

## Script

0.7:

```python
from glotaran.io import load_dataset, load_model, load_parameters, save_result
from glotaran.optimization.optimize import optimize
from glotaran.project.scheme import Scheme

model = load_model("my_model.yaml")
parameters = load_parameters("my_parameters.yaml")
dataset = load_dataset("data.nc")

result = optimize(Scheme(model, parameters, {"my_data": dataset}))
save_result(result, "my_result/result.yml")
```

0.8:

```python
from glotaran.io import load_dataset, load_parameters, load_scheme

scheme = load_scheme("my_scheme.yaml")
parameters = load_parameters("my_parameters.yaml")
dataset = load_dataset("data.nc")

result = scheme.optimize(parameters, {"my_data": dataset})
result.save("my_result")
```

The scheme holds only the model. The parameters, the data and the optimizer settings, such as `maximum_number_function_evaluations`, are arguments of `scheme.optimize`.

## Results

In 0.7, `result.data["my_data"]` was one dataset with every array. In 0.8, `result.optimization_results["my_data"]` has one dataset for each element and each activation. With `fit = result.optimization_results["my_data"]`:

| 0.7: `result.data["my_data"]` | 0.8 | Dimensions in 0.8 |
| --- | --- | --- |
| `.species_associated_spectra` | `fit.elements["sequential"].amplitudes` | `spectral`, `compartment` |
| `.decay_associated_spectra_m1` | `fit.elements["sequential"].kinetic_amplitudes` | `activation`, `spectral`, `kinetic` |
| `.species_concentration` | `fit.elements["sequential"].concentrations` | `time`, `compartment` |
| `.a_matrix_m1`, `.k_matrix_m1`, `.k_matrix_reduced_m1` | `fit.elements["sequential"]` with `a_matrix`, `k_matrix`, `reduced_k_matrix` | |
| `.irf`, `.irf_center` | `fit.activations["irf"]` with `trace`, `center` and `shift` | |
| `.fitted_data` | `fit.fitted_data` | `time`, `spectral` |
| `.residual` | `fit.residuals` | `time`, `spectral` |
| `.data` | `fit.input_data` | |
| `.clp`, `.matrix` | `fit.fit_decomposition.clp`, `fit.fit_decomposition.matrix` | |

The 0.7 names end in the label of the megacomplex (`_m1`); the 0.8 names are the same for every element, because each element has its own dataset. The dimension `species` is now `compartment`.

The singular value decomposition of the data and the residuals is not part of a 0.8 result. `add_svd_to_dataset` from `glotaran.io.prepare_dataset` adds it to a dataset.

The fit statistics moved from the result to `result.optimization_info`: `success`, `termination_reason`, `root_mean_square_error`, `chi_square`, `covariance_matrix`, `jacobian` and the others listed under [Results](reference/public_api.md#results) in the reference.

## Removed in 0.8

- The `glotaran` command line interface.
- The 0.7 `Project` with its registries of models, parameters and data, and with `generate_model` and `generate_parameters`. The 0.8 [Project](user_documentation/project.md) records and exports fits.
- `Result.markdown()` and the file `result.md` in a saved result.

## Plotting

pyglotaran-extras 0.7 reads only 0.7 results. For 0.8 results, install pyglotaran-extras as described under [Installation](installation.md), and plot as in the {doc}`getting started guide <notebooks/getting_started/getting_started>`:

```python
from pyglotaran_extras import plot_overview
from pyglotaran_extras.compat import convert

plot_overview(convert(result))
```

## Plugins

A megacomplex plugin becomes an element plugin: a subclass of `Element` from `glotaran.model.element`, registered under the entry point group `glotaran.plugins.elements`.

If a 0.7 analysis does not port, or gives a different answer in 0.8, open an [issue](https://github.com/glotaran/pyglotaran/issues).
