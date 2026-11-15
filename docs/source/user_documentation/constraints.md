# Constraints, relations and penalties

In a fit, the amplitudes of each compartment along the global dimension (the spectra) are free: every wavelength gets its own value. Three items in the scheme restrict them with knowledge about the sample.

These items apply only to datasets with free amplitudes. A dataset with `global_elements` models the global dimension with parameters, has no free amplitudes, and ignores constraints, relations and penalties without a message.

| Item | Where in the scheme | Effect |
| --- | --- | --- |
| Constraint of type `zero` | `clp_constraints` of an element | The spectrum of `target` is zero inside `interval` |
| Constraint of type `only` | `clp_constraints` of an element | The spectrum of `target` is zero outside `interval` |
| Relation | `clp_relations` of an experiment | Inside `interval`, the spectrum of `target` is `parameter` times the spectrum of `source` |
| Penalty of type `equal_area` | `clp_penalties` of an experiment | Adds a term to the residual that grows with the difference between the area of `source` and `parameter` times the area of `target` |

An interval is `[minimum, maximum]` on the global dimension, and a list of intervals is allowed. Without an interval the item applies everywhere. `parameter` is a number or the label of a parameter, which is then optimized like any other.

## Example

The data of this script are simulated so that all three statements are true: the spectrum of `s1` is zero above 660, up to 612 the spectrum of `s2` is half that of `s1`, and the areas of the spectra have the ratio 0.989.

Script: {download}`constraints.py <../examples/constraints.py>`

```{literalinclude} ../examples/constraints.py
:language: python
:start-at: import numpy
```

The script prints:

```text
simulated: s1 is zero above 660, s2 / s1 is 0.5 up to 612, area ratio 0.989
without constraints
  rates: [0.602, 0.1]
  largest amplitude of s1 above 660: 0.0202
  s2 / s1 up to 612: -0.870 to 1.733
  area of s1 / area of s2: 0.990
zero constraint
  rates: [0.602, 0.1]
  largest amplitude of s1 above 660: 0.0000
  s2 / s1 up to 612: -0.870 to 1.749
  area of s1 / area of s2: 0.983
relation
  rates: [0.602, 0.1]
  largest amplitude of s1 above 660: 0.0202
  s2 / s1 up to 612: 0.500 to 0.500
  area of s1 / area of s2: 0.992
equal area penalty
  rates: [0.602, 0.1]
  largest amplitude of s1 above 660: 0.0201
  s2 / s1 up to 612: -0.870 to 1.735
  area of s1 / area of s2: 0.990
```

The zero constraint and the relation hold exactly in the fitted spectra. Without them, the noise shows in the regions where a spectrum is small.

## The equal area penalty

The penalty adds `weight * |area(source) - parameter * area(target)|` to the residual, where the area is the sum of the amplitudes over `source_intervals` and `target_intervals`, or over the whole global dimension. A larger `weight` enforces the ratio more strictly.

In the example the data determine both spectra, so the penalty changes little. It is used when the relative size of two spectra is not determined by the data. The analysis `ex_spectral_constraints` in [pyglotaran-examples](https://github.com/glotaran/pyglotaran-examples) shows such a case.

## Non-negative amplitudes

To keep the amplitudes that the fit estimates at zero or above, set `residual_function: non_negative_least_squares` on the experiment:

```yaml
experiments:
  my_experiment:
    residual_function: non_negative_least_squares
    datasets:
      my_data:
        elements: [decay]
```

A dataset also accepts `residual_function`, but the fit does not use it.

The amplitudes of the `target` of a relation are not estimated; they are computed afterwards as `parameter` times the amplitudes of `source`. With a negative `parameter`, they are negative also with non-negative least squares.
