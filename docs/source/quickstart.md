# Quickstart

This page runs one fit from a script: it creates a dataset, describes a model, fits it, saves the result and loads it again. The script needs only pyglotaran, without a notebook or a plotting package. The {doc}`getting started guide <notebooks/getting_started/getting_started>` covers the same steps in a notebook, with plots.

## Run the script

Download {download}`fit_and_reload.py <quickstart/fit_and_reload.py>` and run it with [uv]:

```shell
uv run fit_and_reload.py
```

uv reads the dependencies from the comment at the top of the script and creates an environment for it. In an environment that has pyglotaran [installed](installation.md), run it with `python fit_and_reload.py`.

The script prints:

```text
success: True
termination: `ftol` termination condition is satisfied.
root mean square error: 0.0100
rates.1 = 0.5999 +/- 0.0024
rates.2 = 0.0995 +/- 0.0002
loaded rates: [0.5999, 0.0995]
arrays of the element 'decay': ['a_matrix', 'amplitudes', 'concentrations', 'initial_concentrations', 'k_matrix', 'kinetic_amplitudes', 'reduced_k_matrix']
shape of the fitted data: (201, 31)
```

The fit recovers the rates 0.6 and 0.1 that the data were created with.

## The script

```{literalinclude} quickstart/fit_and_reload.py
:language: python
```

## What the steps do

1. **Data.** A dataset is an `xarray.Dataset` with a variable `data` of two dimensions. The kinetic element of the model describes the dimension `time`; the other dimension can have any name. `load_dataset` from `glotaran.io` reads a dataset from a file.
2. **Scheme.** The `library` holds the elements of the model, here one kinetic element with two compartments that each decay with their own rate. The `experiments` assign elements to datasets. The dataset label `my_data` is the key under which the data are passed to `optimize`.
3. **Parameters.** `rates: [1.0, 0.05]` defines the parameters `rates.1` and `rates.2` with their starting values. The scheme refers to parameters by these labels.
4. **Fit.** `scheme.optimize` returns a result. `optimization_info.success` says that the optimizer returned a solution and `termination_reason` says why it stopped; a fit that stops at the limit of function evaluations also has `success == True`. With the default `verbose=True` the fit prints the iteration table of the optimizer.
5. **Save and load.** `save_result` writes the folder `quickstart_result` with the scheme as YAML, the parameters as CSV and the arrays as NetCDF. `load_result` reads the folder back.

## Where to go next

- The {doc}`getting started guide <notebooks/getting_started/getting_started>` adds an instrument response function, plotting and a project that records each fit.
- [Public API and file formats](user_documentation/public_api.md) lists the names, arguments and files used here, for applications that call pyglotaran.

[uv]: https://docs.astral.sh/uv/
