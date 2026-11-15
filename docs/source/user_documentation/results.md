# Read, save and load results

A fit returns a `Result`. This page shows where its parts are and how to save and read them. The complete list of attributes is under [Results](../reference/public_api.md#results) in the reference.

The script starts with the fit of the [quickstart](../quickstart.md), which gives `result`; the part shown here follows it. It uses `numpy` as `np`, `pandas` as `pd`, `xarray` as `xr`, and `SAVING_OPTIONS_MINIMAL` and `load_result` from `glotaran.io`.

Script: {download}`results.py <../examples/results.py>`

```{literalinclude} ../examples/results.py
:language: python
:start-at: "# Parameters: a table"
```

The script prints:

```text
  label  value  standard_error
rates.1 0.5999          0.0024
rates.2 0.0995          0.0002
free parameters: ['rates.1', 'rates.2']
correlation of the two rates: 0.028
spectra: {'spectral': 31, 'compartment': 2}
concentrations: {'time': 201, 'compartment': 2}
residuals: {'time': 201, 'spectral': 31} largest: 0.038
meta: time spectral
loaded: ['a_matrix', 'amplitudes', 'concentrations', 'initial_concentrations', 'k_matrix', 'kinetic_amplitudes', 'reduced_k_matrix']
decay.nc: {'spectral': 31, 'compartment': 2}
optimized_parameters.csv: ['label', 'value', 'standard_error', 'expression', 'maximum', 'minimum', 'non_negative', 'vary']
```

## Where things are

| You want | Where |
| --- | --- |
| Optimized parameters and standard errors | `result.optimized_parameters` |
| Fit statistics and the reason the fit stopped | `result.optimization_info` |
| Covariance matrix of the free parameters | `result.optimization_info.covariance_matrix`, in the order of `free_parameter_labels` |
| Arrays of one dataset | `fit = result.optimization_results["my_data"]` |
| Fitted data and residuals | `fit.fitted_data`, `fit.residuals` |
| Results of one element | `fit.elements["<element label>"]` |
| Results of one activation, such as the IRF | `fit.activations["<activation label>"]` |
| Root mean square error of one dataset | `fit.meta.root_mean_square_error` |

## Kinetic element

The dataset of a kinetic element holds:

| Variable | Dimensions | Content |
| --- | --- | --- |
| `amplitudes` | global dimension, `compartment` | Species associated spectra (SAS) |
| `concentrations` | `time`, `compartment` | Concentration of each compartment |
| `kinetic_amplitudes` | `activation`, global dimension, `kinetic` | Decay associated spectra (DAS) |
| `k_matrix`, `reduced_k_matrix`, `a_matrix`, `initial_concentrations` | | Matrices of the kinetic scheme |

[Result arrays](../reference/results.md) in the reference lists the arrays of every element type and activation.

## Save and load

`result.save("folder")` writes a folder with the scheme, the parameters as CSV and the arrays as NetCDF. `load_result("folder")` reads it back. [Result folder](../reference/public_api.md#result-folder) in the reference lists the files.

- `save` refuses to replace an existing result unless `allow_overwrite=True`.
- `saving_options=SAVING_OPTIONS_MINIMAL` writes the parameters and the statistics without the result arrays. The reference describes what it does with the input data.
- The covariance matrix is not saved. The standard errors are in `optimized_parameters.csv`.

The files are in open formats, so other tools read them without pyglotaran: the arrays with any NetCDF library, the parameters as CSV, and `result.yml` as YAML.

To keep a record of every fit of an analysis, and to export results with their input data, use a [project](project.md).
