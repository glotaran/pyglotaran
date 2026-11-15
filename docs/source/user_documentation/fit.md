# Run a fit and judge the outcome

`scheme.optimize(parameters, datasets)` runs the fit and returns a result. `datasets` maps each dataset label of the scheme to its data. The arguments are listed under [Scheme and fit](../reference/public_api.md#scheme-and-fit) in the reference.

## How a fit can end

`result.optimization_info.success` is `True` when the optimizer returned a solution. It is also `True` when the fit stopped at the limit of function evaluations, so read `termination_reason` as well. This script runs the four cases and tells them apart. It starts with the dataset, the scheme and the parameters of the [quickstart](../quickstart.md); the part shown here follows them.

Script: {download}`fit_outcome.py <../examples/fit_outcome.py>`

```{literalinclude} ../examples/fit_outcome.py
:language: python
:start-at: def outcome
```

The script prints:

```text
fit: converged
  success=True, evaluations=5, rmse=0.0100
  termination_reason: `ftol` termination condition is satisfied.
fit with two evaluations: stopped at the evaluation limit
  success=True, evaluations=2, rmse=0.0115
  termination_reason: The maximum number of function evaluations is exceeded.
dry run: evaluated only
  success=False, evaluations=1, rmse=0.0433
  termination_reason: Dry run.
fit with a starting value outside its bounds: failed
  success=False, evaluations=0, rmse=none
  termination_reason: Initial guess is outside of provided bounds
  warnings: ['Optimization failed:']
  with raise_exception=True: ValueError
```

| Outcome | `success` | `termination_reason` |
| --- | --- | --- |
| Converged | `True` | A termination condition of SciPy (`ftol`, `xtol` or `gtol`) is satisfied |
| Stopped at the evaluation limit | `True` | `The maximum number of function evaluations is exceeded.` |
| No free parameters (every parameter fixed) | `True` | `No free parameters to optimize.` |
| Dry run (`dry_run=True`) | `False` | `Dry run.` |
| Failed | `False` | The message of the error |

A fit without free parameters evaluates the model once; `success` is `True` although nothing was optimized.

An error inside the optimizer gives a warning, and the fit still returns a result. With `raise_exception=True` the error is raised. Some errors are raised whatever the value of `raise_exception`:

- Problems with the input are raised before the fit starts, for example `GlotaranUserError` for a dataset label of the scheme without data, or `ParameterNotFoundError` for a parameter the scheme uses that is missing.
- If the model cannot be evaluated at the parameters where the optimizer stopped, the error of the optimizer is raised after the warning. Data with a NaN in an experiment with `residual_function: non_negative_least_squares` end this way with a `ValueError`.

A script or application that must continue after a failed fit catches exceptions around `scheme.optimize`. [Errors and warnings](../reference/public_api.md#errors-and-warnings) in the reference lists them.

## What to check after a fit

- **Termination.** A fit that stopped at the evaluation limit has not converged. Continue from `result.optimized_parameters`, or raise `maximum_number_function_evaluations`.
- **Root mean square error.** For data without weights, `optimization_info.root_mean_square_error` is in the units of the data. For a model that describes the data it is close to the noise level: 0.0100 above, for noise with a standard deviation of 0.01.
- **Standard errors.** `result.optimized_parameters` has a `standard_error` for each free parameter. A standard error that is large compared with the value means the data do not determine that parameter. For a parameter with `non-negative: true` the fit optimizes the logarithm of the value, and `standard_error` is the error of that logarithm: it is a relative error, and the error of the value is approximately `value * standard_error`. A rate of 0.0995 with a `standard_error` of 0.0022 has an error of about 0.0002.
- **Parameters at a bound.** A value equal to its `minimum` or `maximum` means the bound decides the value.
- **Residuals.** `result.optimization_results["my_data"].residuals` should look like noise. Structure in the residuals means the model misses a component. [Plotting](plotting.md) shows how to look at them.

## A dry run

`dry_run=True` evaluates the model once at the starting values without optimizing. The result has all arrays, so you can compare the starting point with the data before a long fit.

## Output and running fits from an application

With the default `verbose=True` the fit prints the iteration table of the optimizer. `verbose=False` prints nothing.

`scheme.optimize` returns when the fit has ended, and replaces `sys.stdout` of the process while the optimizer runs. An application that needs a timeout or a way to cancel a fit runs the fit in a separate Python process.
