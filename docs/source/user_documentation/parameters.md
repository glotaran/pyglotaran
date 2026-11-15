# Parameters

Parameters hold the starting values of a fit. The scheme refers to a parameter by its label.

## Define parameters

A YAML parameter file is a mapping of groups. A group is a list of parameters, and a parameter is a value, or a list with an optional label, the value and an optional mapping of options.

Script: {download}`parameters.py <../examples/parameters.py>`

```{literalinclude} ../examples/parameters.py
:language: python
:start-at: from glotaran.io import load_parameters
```

The script prints:

```text
       label  value                expression  minimum  maximum  non_negative  vary
  rates.fast    2.0                       NaN     0.00      inf         False  True
  rates.slow    0.1                       NaN     -inf      inf          True  True
 rates.total    2.1 $rates.fast + $rates.slow     -inf      inf         False False
  irf.center    0.5                       NaN     -inf      inf         False False
   irf.width    0.1                       NaN     0.01      1.0         False  True
amplitudes.1    1.0                       NaN     -inf      inf         False  True
amplitudes.2    0.5                       NaN     -inf      inf         False  True
irf.width: 0.1 0.01 1.0 True
reloaded labels: ['amplitudes.1', 'amplitudes.2', 'irf.center', 'irf.width', 'rates.fast', 'rates.slow', 'rates.total']
rates.fast: 3.0
amplitudes.2 varies: False
rates.total: 3.1 = $rates.fast + $rates.slow
```

## Labels

The label of a parameter is the label of its group and its own label, joined by `.`: `rates.fast`, `irf.center`. A parameter without a label gets its position in the list, counted from 1: `amplitudes.1`, `amplitudes.2`. A label consists of ASCII letters, digits, `_` and `.`.

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `vary` | Optimize the parameter. With `false` it keeps its value | `true` |
| `min`, `max` | Bounds of the value | none |
| `non-negative` | Keep the value positive by optimizing its logarithm | `false` |
| `expr` | Compute the value from other parameters, referred to as `$label` | none |

A parameter with an expression is not optimized; its value follows the parameters it refers to, as `rates.total` does after `rates.fast` is changed.

A mapping of options as the last item of a group applies to every parameter of the group:

```yaml
rates:
  - [fast, 2.0]
  - [slow, 0.1]
  - { non-negative: true }
```

```{warning}
An option that is not in this table is ignored without a message. With a misspelled option, such as `{vari: false}`, the parameter stays free.
```

## Parameters in code

`parameters.get(label)` returns one parameter with the fields `value`, `minimum`, `maximum`, `vary`, `non_negative`, `expression` and, after a fit, `standard_error`. `parameters.to_dataframe()` returns all of them as a table. Changing `value` or `vary` before a fit changes the starting point; `scheme.optimize` copies the parameters and returns the optimized ones in `result.optimized_parameters`.

A fit ignores parameters that the scheme does not use.

## Files

| Format | Load | Save |
| --- | --- | --- |
| `yml`, `yaml` | yes | no |
| `csv`, `tsv`, `xlsx`, `ods` | yes | yes |

A table file has one row for each parameter with the columns `label` and `value`, and optionally `standard_error`, `expression`, `minimum`, `maximum`, `non_negative` and `vary`. To continue from a fit, save `result.optimized_parameters` as a table and load it as the starting values of the next fit.
