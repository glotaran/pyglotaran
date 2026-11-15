# Plotting

pyglotaran computes; the plotting functions are in the companion package [pyglotaran-extras](https://pyglotaran-extras.readthedocs.io). [Installation](../installation.md) shows how to install the version that works with pyglotaran 0.8.

```python
from pyglotaran_extras import plot_data_overview, plot_overview
from pyglotaran_extras.compat import convert

plot_data_overview(dataset)      # the data before a fit
plot_overview(convert(result))   # data, fit, residuals, spectra and concentrations
```

The {doc}`getting started guide <../notebooks/getting_started/getting_started>` uses both functions and explains the panels of the overview.

The arrays of a result are `xarray` objects, so their own plotting works without pyglotaran-extras:

```python
fit = result.optimization_results["my_data"]

fit.elements["decay"].amplitudes.plot.line(x="spectral")   # spectra
fit.elements["decay"].concentrations.plot.line(x="time")   # concentrations
fit.residuals.plot()                                       # residuals as a map
```

These lines need matplotlib, which pyglotaran does not install.
