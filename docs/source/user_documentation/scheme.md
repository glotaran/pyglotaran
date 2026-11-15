# Write a scheme

A scheme describes the model of an analysis. It has two parts:

- `library` holds the **elements**: the building blocks of the model, such as a kinetic scheme.
- `experiments` assign elements to **datasets**. The label of a dataset is the key under which its data are passed to `scheme.optimize`.

A scheme is a YAML file, loaded with `load_scheme("scheme.yml")`. It holds no parameter values, data or optimizer settings.

## A sequential model with an instrument response function

In this script, compartment `s1` is excited through a Gaussian instrument response function (IRF), converts to `s2`, and `s2` decays. The script simulates data for this model and fits them.

Script: {download}`target_model.py <../examples/target_model.py>`

```{literalinclude} ../examples/target_model.py
:language: python
:start-at: import numpy
```

The script prints:

```text
termination: `ftol` termination condition is satisfied.
rates.1 = 2.0001 +/- 0.0046
rates.2 = 0.1000 +/- 0.0001
irf.center = 0.4996 +/- 0.0004
irf.width = 0.0993 +/- 0.0006
species associated spectra: {'spectral': 35, 'compartment': 2}
spectral maxima: {'s1': 620.0, 's2': 660.0}
```

The fit recovers the rates 2.0 and 0.1, the IRF and the positions of the two spectra.

## Kinetic element

An element of `type: kinetic` has `rates`. A key `(to, from)` names the target and the source compartment, and the value is a parameter label or a number.

| Rates | Meaning |
| --- | --- |
| `(s1, s1): k` | `s1` decays with rate `k` |
| `(s2, s1): k` | `s1` converts to `s2` with rate `k` |

Models are combinations of these:

```yaml
library:
  parallel: # s1 and s2 decay independently
    type: kinetic
    rates:
      (s1, s1): rates.1
      (s2, s2): rates.2

  sequential: # s1 -> s2 -> s3, and s3 decays
    type: kinetic
    rates:
      (s2, s1): rates.1
      (s3, s2): rates.2
      (s3, s3): rates.3

  branched: # s1 converts to s2 and to s3
    type: kinetic
    rates:
      (s2, s1): rates.1
      (s3, s1): rates.2
      (s2, s2): rates.3
      (s3, s3): rates.4
```

A kinetic element can take over the rates of other kinetic elements with `extends: [other]` and add or replace rates.

## Activations

An activation says how the compartments of the kinetic elements of a dataset are excited. `compartments` gives the input of each excited compartment as a parameter label or a number; a compartment that is not listed gets no input.

| Type | Excitation | Fields |
| --- | --- | --- |
| `instant` | At time 0, without an instrument response | `compartments` |
| `gaussian` | Through a Gaussian IRF | `compartments`, `center`, `width` |
| `multi-gaussian` | Through an IRF that is a sum of Gaussians | `compartments`, and lists for `center` and `width`, with an optional `scale` |

The Gaussian activations have optional fields for a dispersion of the IRF along the global dimension (`dispersion_center`, `center_dispersion_coefficients`, `width_dispersion_coefficients`), for a `shift` of the center and for the `backsweep` of a streak camera.

## Other elements

| Type | Describes | Fields |
| --- | --- | --- |
| `spectral` | Spectral shapes along the dimension `spectral` | `shapes`, each of type `gaussian`, `skewed-gaussian`, `one` or `zero` |
| `baseline` | A constant offset | `dimension` |
| `coherent-artifact` | A coherent artifact around time zero | `order`, `width`; the dataset needs `activations` |
| `damped-oscillation` | Damped oscillations | `oscillations`, each with `frequency` and `rate`; the dataset needs `activations` |
| `pfid` | Perturbed free induction decay | `oscillations`, each with `frequency` and `rate`; the dataset needs one Gaussian `activation` |
| `clp-guide` | Guidance of a spectrum by a known one | `target`, `dimension` |

Every element has a `dimension`: the model dimension it describes. The kinetic, coherent-artifact, damped-oscillation and pfid elements default to `time` and the spectral element to `spectral`. The baseline and clp-guide elements have no default, so set their `dimension`, for example `dimension: time` next to a kinetic element. All elements of a dataset need the same dimension.

A dataset can combine elements, for example `elements: [sequential, artifact, offset]`. The analyses in [pyglotaran-examples](https://github.com/glotaran/pyglotaran-examples) use these elements on measured data.

In the simulation above, `global_elements: [spectra]` gives the dataset a model for the global dimension, so that the spectra follow from parameters. In the fit the dataset has no global elements, and every wavelength gets its own amplitudes.

## Residual function

An experiment can set `residual_function`. The default `variable_projection` allows positive and negative amplitudes; `non_negative_least_squares` keeps all amplitudes of the experiment at zero or above. A dataset also accepts `residual_function`, but the fit does not use it; set it on the experiment.

## Completion and validation in an editor

`create_model_scheme_json_schema` writes the JSON Schema of a scheme for the installed version:

```python
from glotaran.utils.json_schema import create_model_scheme_json_schema

create_model_scheme_json_schema("schema.json")
```

An editor with YAML language support, such as VS Code with the YAML extension, uses the schema when the scheme file starts with this line:

```yaml
# yaml-language-server: $schema=schema.json
```

The schema checks field names and types, which covers part of what `load_scheme` checks. It does not check that a parameter label or an element label exists, or that a dataset has the fields its elements need, such as `activations` for a kinetic element. pyglotaran reports these problems when the scheme is loaded, and a missing parameter when the fit starts.
