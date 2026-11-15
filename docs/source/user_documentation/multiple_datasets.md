# Fit several datasets together

A scheme can hold several datasets. Fitted together, they share the parameters that their elements refer to.

## Datasets in one experiment

Datasets in the same experiment share the parameters and also the amplitudes: at each value of the global dimension, the datasets get the same spectra. Use this for measurements of the same sample, for example with a short and a long time range.

Script: {download}`two_datasets.py <../examples/two_datasets.py>`

```{literalinclude} ../examples/two_datasets.py
:language: python
:start-at: import numpy
```

The script prints:

```text
termination: `ftol` termination condition is satisfied.
rates.1 = 0.5971 +/- 0.0028
rates.2 = 0.0995 +/- 0.0004
scale.long = 0.4997 +/- 0.0009
short: root mean square error 0.0099
long: root mean square error 0.0100
same spectra in both datasets: True
```

Each dataset has its own entry in `result.optimization_results`, with its own residuals and root mean square error.

## Scale

Measurements of the same sample often differ in signal size. `scale` of the experiment gives a dataset a factor relative to the others; a dataset that is not listed has the factor 1. Above, the fit finds 0.4997 for the dataset that was simulated with half the signal.

Without `scale`, the same two datasets do not fit with shared spectra: the root mean square error is 0.054 where the noise is 0.01, and the rates come out as 0.58 and 0.18 for simulated 0.6 and 0.1.

## Datasets in separate experiments

Datasets in different experiments share only the parameters. Each dataset gets its own spectra, so no `scale` is needed. Use this for different samples, or different kinds of measurement, that share rates.

```yaml
experiments:
  first:
    datasets:
      short:
        elements: [decay]
        activations:
          excitation:
            type: instant
            compartments: {s1: 1, s2: 1}
  second:
    datasets:
      long:
        elements: [decay]
        activations:
          excitation:
            type: instant
            compartments: {s1: 1, s2: 1}
```

With the data of the script above, this scheme gives the rates 0.5987 and 0.0997, and the spectra of `long` are half those of `short`.

## Global dimensions that differ

The amplitudes of datasets in one experiment are shared where the values of their global dimension match. `clp_link_tolerance` of the experiment sets how far two values can be apart to count as the same (default 0), and `clp_link_method` how a value is matched: `nearest` (default), `backward` or `forward`.

Dataset labels are unique across the experiments of a scheme.
