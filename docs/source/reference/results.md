# Result arrays

This page lists the arrays that each element and activation adds to a result. With `fit = result.optimization_results["<dataset label>"]`, every element of the dataset has an `xarray.Dataset` in `fit.elements["<element label>"]` and every Gaussian activation one in `fit.activations["<activation label>"]`. [Read, save and load results](../user_documentation/results.md) shows how to use them.

In the tables, **model** is the model dimension of the dataset, for example `time`, and **global** is its global dimension, for example `spectral`.

## Elements

### kinetic

| Variable | Dimensions | Content |
| --- | --- | --- |
| `amplitudes` | global, `compartment` | Amplitude of each compartment: the species associated spectra (SAS) |
| `concentrations` | model, `compartment` | Concentration of each compartment. With an activation that depends on the global dimension (`shift` or a dispersion) the dimensions are global, model, `compartment` |
| `kinetic_amplitudes` | `activation`, global, `kinetic` | Amplitude of each decay rate: the decay associated spectra (DAS) |
| `initial_concentrations` | `activation`, `compartment` | Input of each compartment after normalization |
| `k_matrix` | `to`, `from` | The matrix of the rate equations: an off-diagonal entry is the rate from `from` to `to`, and a diagonal entry is minus the sum of the rates that leave the compartment |
| `reduced_k_matrix` | `to`, `from` | The rates as written in the scheme: the entry at `to`, `from` is the rate with that key |
| `a_matrix` | `activation`, `compartment`, `kinetic` | Contribution of each decay rate to each compartment |

The dimension `kinetic` has the coordinates `rate` and `lifetime`.

### spectral

| Variable | Dimensions | Content |
| --- | --- | --- |
| `amplitudes` | global, `shape` | Amplitude of each shape |
| `concentrations` | model, `shape` | The shapes |

### baseline

| Variable | Dimensions | Content |
| --- | --- | --- |
| `amplitudes` | global | The offset |

### coherent-artifact

| Variable | Dimensions | Content |
| --- | --- | --- |
| `amplitudes` | global, `derivative` | Amplitude of each term; `derivative` counts from 0, the Gaussian itself |
| `concentrations` | model, `derivative` | The terms along the model dimension |

### damped-oscillation

| Variable | Dimensions | Content |
| --- | --- | --- |
| `cos_amplitudes`, `sin_amplitudes` | global, `oscillation` | Amplitudes of the cosine and the sine term of each oscillation |
| `amplitudes` | global, `oscillation` | `sqrt(sin_amplitudes**2 + cos_amplitudes**2)` |
| `phase_amplitudes` | global, `oscillation` | Phase from the sine and cosine amplitudes, unwrapped along the global dimension |
| `cos_concentrations`, `sin_concentrations` | model, `oscillation` | The cosine and the sine term along the model dimension |
| `concentrations` | model, `oscillation` | `sqrt(sin_concentrations**2 + cos_concentrations**2)` |
| `phase_concentrations` | model, `oscillation` | Phase from the sine and cosine terms |

The dimension `oscillation` has the coordinates `oscillation_frequency` and `oscillation_rate`.

### pfid

The variables are those of `damped-oscillation`, with two differences: the phase of the amplitudes is named `phase`, and the variables ending in `concentrations` have the dimensions global, model, `oscillation`.

### clp-guide

The dataset of a `clp-guide` element is empty. The guided amplitude is in `fit.fit_decomposition.clp`.

## Activations

A `gaussian` or `multi-gaussian` activation has a dataset; an `instant` activation has none.

| Variable | Dimensions | Content |
| --- | --- | --- |
| `trace` | model | The instrument response function |
| `center` | `component_index`, global | Center of each Gaussian at each value of the global dimension |
| `shift` | global | Shift of the center at each value of the global dimension |

## Fit decomposition

`fit.fit_decomposition` holds the two factors of the fit of one dataset.

| Variable | Dimensions | Content |
| --- | --- | --- |
| `clp` | global, `amplitude_label` | The conditionally linear parameters: all amplitudes of the dataset |
| `matrix` | model, `amplitude_label` | The model matrix. With an activation that depends on the global dimension the dimensions are global, model, `amplitude_label` |

The values of `amplitude_label` come from the elements:

| Element | Labels |
| --- | --- |
| `kinetic` | The compartments |
| `spectral` | The shapes |
| `baseline` | `baseline_<element label>` |
| `coherent-artifact` | `<element label>_derivative_<number>` |
| `damped-oscillation`, `pfid` | `<oscillation label>_cos` and `<oscillation label>_sin` |

Constraints, relations and penalties refer to amplitudes by these labels.

## Datasets with global elements

A dataset with `global_elements` is described in both dimensions, so its amplitudes are not free along the global dimension. Its result has one entry in `fit.elements`, named after the dataset:

| Variable | Dimensions | Content |
| --- | --- | --- |
| `amplitudes` | `global_clp_label`, `clp_label` | Amplitude of each pair of a global and a model component |
| `global_concentrations` | global, `amplitude_label` | The global elements along the global dimension |
| `model_concentrations` | model, `amplitude_label` | The elements along the model dimension |
