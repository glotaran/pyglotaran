..
    Don't change api_documentation.rst since it changes will be overwritten.
    If you want to change api_documentation.rst you have to make the changes in
    api_documentation_template.rst and run `make api_docs` afterwards.
    For changes to take effect you might also have to run `make clean_all`
    afterwards.

API Documentation
=================

The API Documentation for glotaran is automatically created from its docstrings.

.. currentmodule:: glotaran

.. autosummary::
    :toctree: api

    glotaran.analysis
    glotaran.analysis.dataset
    glotaran.analysis.fitresult
    glotaran.analysis.fitting
    glotaran.analysis.grouping
    glotaran.analysis.simulation
    glotaran.analysis.test
    glotaran.analysis.test.mock
    glotaran.analysis.test.test_fitting
    glotaran.analysis.test.test_grouping
    glotaran.analysis.test.test_simulation
    glotaran.analysis.variable_projection
    glotaran.dataio
    glotaran.dataio.chlorospec_format
    glotaran.dataio.mlsd_file_format
    glotaran.dataio.spectral_timetrace
    glotaran.dataio.wavelength_time_explicit_file
    glotaran.model
    glotaran.model.base_model
    glotaran.model.compartment_constraints
    glotaran.model.dataset
    glotaran.model.dataset_descriptor
    glotaran.model.model
    glotaran.model.model_item
    glotaran.model.model_item_validator
    glotaran.model.parameter
    glotaran.model.parameter_group
    glotaran.models
    glotaran.models.damped_oscillation
    glotaran.models.damped_oscillation.doas_matrix
    glotaran.models.damped_oscillation.doas_megacomplex
    glotaran.models.damped_oscillation.doas_model
    glotaran.models.damped_oscillation.doas_spectral_matrix
    glotaran.models.damped_oscillation.oscillation
    glotaran.models.spectral_temporal
<<<<<<< HEAD
<<<<<<< HEAD
    glotaran.models.spectral_temporal.c_matrix_cython
    glotaran.models.spectral_temporal.c_matrix_cython.c_matrix_cython
    glotaran.models.spectral_temporal.irf
    glotaran.models.spectral_temporal.irf_gaussian
    glotaran.models.spectral_temporal.irf_measured
=======
    glotaran.models.spectral_temporal.initial_concentration
    glotaran.models.spectral_temporal.irf
>>>>>>> fixed flake8 errors
=======
    glotaran.models.spectral_temporal.initial_concentration
    glotaran.models.spectral_temporal.irf
>>>>>>> c40a2b13b38b7801860ef69cb361f1e037aea3cd
    glotaran.models.spectral_temporal.k_matrix
    glotaran.models.spectral_temporal.kinetic_fitmodel
    glotaran.models.spectral_temporal.kinetic_matrix
    glotaran.models.spectral_temporal.kinetic_megacomplex
    glotaran.models.spectral_temporal.kinetic_model
    glotaran.models.spectral_temporal.kinetic_result
    glotaran.models.spectral_temporal.spectral_matrix
    glotaran.models.spectral_temporal.spectral_shape
<<<<<<< HEAD
<<<<<<< HEAD
    glotaran.models.spectral_temporal.spectral_shape_gaussian
    glotaran.models.spectral_temporal.spectral_temporal_dataset
    glotaran.models.spectral_temporal.spectral_temporal_dataset_descriptor
    glotaran.plotting
    glotaran.plotting.basic_plots
    glotaran.plotting.glotaran_color_codes
    glotaran.specification_parser
    glotaran.specification_parser.model_spec_yaml
    glotaran.specification_parser.model_spec_yaml_doas
    glotaran.specification_parser.model_spec_yaml_kinetic
    glotaran.specification_parser.parser
    glotaran.specification_parser.utils
=======
    glotaran.models.spectral_temporal.spectral_temporal_dataset
    glotaran.models.spectral_temporal.spectral_temporal_dataset_descriptor
    glotaran.parse
    glotaran.parse.parser
    glotaran.parse.register
    glotaran.plotting
    glotaran.plotting.basic_plots
    glotaran.plotting.glotaran_color_codes
>>>>>>> fixed flake8 errors
=======
    glotaran.models.spectral_temporal.spectral_temporal_dataset
    glotaran.models.spectral_temporal.spectral_temporal_dataset_descriptor
    glotaran.parse
    glotaran.parse.parser
    glotaran.parse.register
    glotaran.plotting
    glotaran.plotting.basic_plots
    glotaran.plotting.glotaran_color_codes
>>>>>>> c40a2b13b38b7801860ef69cb361f1e037aea3cd
