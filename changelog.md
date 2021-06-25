# Changelog

## 0.4.0 (2021-06-25)

### ✨ Features

- Add basic spectral model (#672)
- Add Channel/Wavelength dependent shift parameter to irf. (#673)
- Refactored Problem class into GroupedProblem and UngroupedProblem (#681)
- Plugin system was rewritten (#600, #665)
- Deprecation framework (#631)
- Better notebook integration (689)

### 🩹 Bug fixes

- Fix excessive memory usage in `_create_svd` (#576)
- Fix several issues with KineticImage model (#612)
- Fix exception in sdt reader index calculation (#647)
- Avoid crash in result markdown printing when optimization fails (#630)
- ParameterNotFoundException doesn't prepend '.' if path is empty (#688)
- Ensure Parameter.label is str or None (#678)
- Properly scale StdError of estimated parameters with RMSE (#704)
- More robust covariance_matrix calculation (#706)
- `ParameterGroup.markdown()` independent parametergroups of order (#592)

### 🔌 Plugins

- `ProjectIo` 'folder'/'legacy' plugin to save results (#620)
- `Model` 'spectral-model' (#672)

### 📚 Documentation

- User documentation is written in notebooks (#568)
- Documentation on how to write a `DataIo` plugin (#600)

### 🗑️ Deprecations (due in 0.6.0)

- `glotaran.ParameterGroup` -> `glotaran.parameterParameterGroup`
- `glotaran.read_model_from_yaml` -> `glotaran.io.load_model(..., format_name="yaml_str")`
- `glotaran.read_model_from_yaml_file` -> `glotaran.io.load_model(..., format_name="yaml")`
- `glotaran.read_parameters_from_csv_file` -> `glotaran.io.load_parameters(..., format_name="csv")`
- `glotaran.read_parameters_from_yaml` -> `glotaran.io.load_parameters(..., format_name="yaml_str")`
- `glotaran.read_parameters_from_yaml_file` -> `glotaran.io.load_parameters(..., format_name="yaml")`
- `glotaran.io.read_data_file` -> `glotaran.io.load_dataset`
- `result.save` -> `glotaran.io.save_result(result, ..., format_name="legacy")`
- `result.get_dataset("<dataset_name>")` -> `result.data["<dataset_name>"]`
- `glotaran.analysis.result` -> `glotaran.project.result`
- `glotaran.analysis.scheme` -> `glotaran.project.scheme`
- `model.simulate` -> `glotaran.analysis.simulation.simulate(model, ...)`

## 0.3.3 (2021-03-18)

- Force recalculation of SVD attributes in `scheme._prepare_data`
  (#597)
- Remove unneeded check in `spectral_penalties._get_area` Fixes (#598)
- Added python 3.9 support (#450)

## 0.3.2 (2021-02-28)

- Re-release of version 0.3.1 due to packaging issue

## 0.3.1 (2021-02-28)

- Added compatibility for numpy 1.20 and raised minimum required numpy
  version to 1.20 (#555)
- Fixed excessive memory consumption in result creation due to full
  SVD computation (#574)
- Added feature parameter history (#557)
- Moved setup logic to `setup.cfg` (#560)

## 0.3.0 (2021-02-11)

- Significant code refactor with small API changes to parameter
  relation specification (see docs)
- Replaced lmfit with scipy.optimize

## 0.2.0 (2020-12-02)

- Large refactor with significant improvements but also small API
  changes (see docs)
- Removed doas plugin

## 0.1.0 (2020-07-14)

- Package was renamed to `pyglotaran` on PyPi

## 0.0.8 (2018-08-07)

- Changed `nan_policiy` to `omit`

## 0.0.7 (2018-08-07)

- Added support for multiple shapes per compartement.

## 0.0.6 (2018-08-07)

- First release on PyPI, support for Windows installs added.
- Pre-Alpha Development
