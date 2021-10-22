# Changelog

## 0.5.0 (2021-10-24)

### ✨ Features
- ✨ Feature: Megacomplex Models (#736)
- ✨ Feature: Full Models (#747)
- ✨ Damped Oscillation Megacomplex (a.k.a. DOAS) (#764)
- ✨ Add Dataset Groups (#851)

### 👌 Minor Improvements:
- 👌 Add dimensions to megacomplex and dataset_descriptor (#702)
- 👌 Improve ordering in k_matrix involved_compartments function (#788)
- 👌 Improvements to application of clp_penalties (equal area) (#801)
- ♻️ Refactor model.from_dict to parse megacomplex_type from dict and add simple_generator for testing (#807)
- ♻️ Refactor model spec (#836)
- ♻️ Refactor Result Saving (#841)

### 🩹 Bug fixes
- 🩹 Fix/cli0.5 (#765)
- 🩹 Fix Performance Regressions (#740)
- 🩹 Fix compartment ordering randomization due to use of set (#799)
- 🩹 Fix check_deprecations not showing deprecation warnings (#775)
- 🩹 Fix and re-enable IRF Dispersion Test (#786)
- 🩹 Fix coherent artifact crash for index dependent models #808
- 🩹 False positive model validation fail when combining multiple default megacomplexes (#797)
- 🩹 Fix ParameterGroup repr when created with 'from_list' (#827)
- 🩹 Fix for DOAS with reversed oscillations (negative rates) (#839)
- 🩹 Fix parameter expression parsing (#843)
- 🩹 Use a context manager when opening a nc dataset (#848)

### 📚 Documentation

- 📚 Moved API documentation from User to Developer Docs (#776)
- 📚 Add docs for the CLI (#784)
- 📚 Fix deprecation in model used in quickstart notebook (#834)

### 🗑️ Deprecations (due in 0.7.0)

- `glotaran.model.Model.model_dimension` -> `glotaran.project.Scheme.model_dimension`
- `glotaran.model.Model.global_dimension` -> `glotaran.project.Scheme.global_dimension`
- `<model_file>.type.kinetic-spectrum` -> `<model_file>.default_megacomplex.decay`
- `<model_file>.type.spectral-model` -> `<model_file>.default_megacomplex.spectral`
- `<model_file>.spectral_relations` -> `<model_file>.clp_relations`
- `<model_file>.spectral_relations.compartment` -> `<model_file>.clp_relations.source`
- `<model_file>.spectral_constraints` -> `<model_file>.clp_constraints`
- `<model_file>.spectral_constraints.compartment` -> `<model_file>.clp_constraints.target`
- `<model_file>.equal_area_penalties` -> `<model_file>.clp_area_penalties`
- `<model_file>.irf.center_dispersion` -> `<model_file>.irf.center_dispersion_coefficients`
- `<model_file>.irf.width_dispersion` -> `<model_file>.irf.width_dispersion_coefficients`
- `glotaran.project.Scheme(..., non_negative_least_squares=...)` -> `<model_file>dataset_groups.default.residual_function`
- `glotaran.project.Scheme(..., group=...)` -> `<model_file>dataset_groups.default.link_clp`
- `glotaran.project.Scheme(..., group_tolerance=...)` -> `glotaran.project.Scheme(..., clp_link_tolerance=...)`

### 🚧 Maintenance
- 🧪🚇 Add integration test result validation (#754)
- 🔧 Add more QA tools for parts of glotaran (#739)
- 🔧 Fix interrogate usage (#781)
- 🚇 Speedup PR benchmark (#785)


## 0.4.0 (2021-06-25)

### ✨ Features

- Add basic spectral model (#672)
- Add Channel/Wavelength dependent shift parameter to irf. (#673)
- Refactored Problem class into GroupedProblem and UngroupedProblem (#681)
- Plugin system was rewritten (#600, #665)
- Deprecation framework (#631)
- Better notebook integration (#689)

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

- Force recalculation of SVD attributes in `scheme._prepare_data` (#597)
- Remove unneeded check in `spectral_penalties._get_area` Fixes (#598)
- Added python 3.9 support (#450)

## 0.3.2 (2021-02-28)

- Re-release of version 0.3.1 due to packaging issue

## 0.3.1 (2021-02-28)

- Added compatibility for numpy 1.20 and raised minimum required numpy version to 1.20 (#555)
- Fixed excessive memory consumption in result creation due to full SVD computation (#574)
- Added feature parameter history (#557)
- Moved setup logic to `setup.cfg` (#560)

## 0.3.0 (2021-02-11)

- Significant code refactor with small API changes to parameter relation specification (see docs)
- Replaced lmfit with scipy.optimize

## 0.2.0 (2020-12-02)

- Large refactor with significant improvements but also small API changes (see docs)
- Removed doas plugin

## 0.1.0 (2020-07-14)

- Package was renamed to `pyglotaran` on PyPi

## 0.0.8 (2018-08-07)

- Changed `nan_policiy` to `omit`

## 0.0.7 (2018-08-07)

- Added support for multiple shapes per compartment.

## 0.0.6 (2018-08-07)

- First release on PyPI, support for Windows installs added.
- Pre-Alpha Development
