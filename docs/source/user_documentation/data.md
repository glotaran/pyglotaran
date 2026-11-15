# Prepare data

A dataset is an `xarray.Dataset` with a variable `data` that has two dimensions:

- The **model dimension** is the one the elements of the scheme describe. Its name is fixed by the element: `time` for a kinetic element.
- The **global dimension** is the other one, for example the wavelength. It can have any name; these pages use `spectral`.

The order of the two dimensions does not matter. An `xarray.DataArray` is accepted as well and is used as `data`.

## Create, weight, save and load a dataset

Script: {download}`data.py <../examples/data.py>`

```{literalinclude} ../examples/data.py
:language: python
:start-at: import numpy
```

The script prints:

```text
from arrays: {'time': 201, 'spectral': 31} ['data']
from a table: {'time': 201, 'spectral': 31}
weights: [0.1, 1.0]
from data_example.nc: {'time': 201, 'spectral': 31} ['data', 'weight']
from data_example.ascii: {'time': 201, 'spectral': 31, 'left_singular_value_index': 31, 'singular_value_index': 31, 'right_singular_value_index': 31} ['data', 'data_left_singular_vectors', 'data_right_singular_vectors', 'data_singular_values']
same values: True
```

## Weights

A variable `weight` with the shape of `data` multiplies the residuals of the fit. A weight below 1 makes a region count less.

Weights can also be set in the scheme, as `weights` of a dataset. Each entry has a `value` and an optional interval on the global dimension, on the model dimension, or on both:

```yaml
experiments:
  my_experiment:
    datasets:
      my_data:
        elements: [decay]
        weights:
          - value: 0.1
            global_interval: [600, 615]
          - value: 0.5
            model_interval: [0, 1]
```

Weights from the scheme are used in the fit and are not added to the `input_data` of the result.

## File formats

| Format | Load | Save | Content |
| --- | --- | --- | --- |
| `nc` | yes | yes | NetCDF as written by xarray; keeps the whole dataset |
| `ascii` | yes | yes | Wavelength-explicit or time-explicit text file with one array. Loading adds the singular value decomposition of the data |
| `sdt` | yes | no | Becker & Hickl SDT file |

`load_dataset` and `save_dataset` take the format from the file extension, or from `format_name`.

## Before you fit

- The data cannot contain `NaN`. A fit of such data ends with `success == False` and the reason `Residuals are not finite in the initial point.`
- The model dimension needs the name the element expects. A kinetic element with data that has no dimension `time` raises `KeyError`.
- The dataset needs the variable `data`, or is a `DataArray`.
