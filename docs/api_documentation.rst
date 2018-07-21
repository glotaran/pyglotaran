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

    glotaran.dataio
    glotaran.dataio.chlorospec_format
    glotaran.dataio.mlsd_file_format
    glotaran.dataio.spectral_timetrace
    glotaran.dataio.wavelength_time_explicit_file
    glotaran.fitmodel
    glotaran.fitmodel.fitmodel
    glotaran.fitmodel.matrix
    glotaran.fitmodel.matrix_group
    glotaran.fitmodel.matrix_group_generator
    glotaran.fitmodel.result
    glotaran.model
    glotaran.model.c_matrix
    glotaran.model.compartment_constraints
    glotaran.model.dataset
    glotaran.model.dataset_descriptor
    glotaran.model.initial_concentration
    glotaran.model.megacomplex
    glotaran.model.model
    glotaran.model.parameter
    glotaran.model.parameter_constraints
    glotaran.model.parameter_group
    glotaran.models
    glotaran.models.spectral_temporal
    glotaran.models.spectral_temporal.c_matrix_cython
    glotaran.models.spectral_temporal.c_matrix_cython.c_matrix_cython
    glotaran.models.spectral_temporal.dataset
    glotaran.models.spectral_temporal.dataset_descriptor
    glotaran.models.spectral_temporal.fitmodel
    glotaran.models.spectral_temporal.irf
    glotaran.models.spectral_temporal.irf_gaussian
    glotaran.models.spectral_temporal.k_matrix
    glotaran.models.spectral_temporal.kinetic_matrix
    glotaran.models.spectral_temporal.megacomplex
    glotaran.models.spectral_temporal.model_kinetic
    glotaran.models.spectral_temporal.result
    glotaran.models.spectral_temporal.spectral_matrix
    glotaran.models.spectral_temporal.spectral_shape
    glotaran.models.spectral_temporal.spectral_shape_gaussian
    glotaran.plotting
    glotaran.plotting.basic_plots
    glotaran.plotting.glotaran_color_codes
    glotaran.specification_parser
    glotaran.specification_parser.model_spec_yaml
    glotaran.specification_parser.model_spec_yaml_kinetic
    glotaran.specification_parser.parser
    glotaran.specification_parser.utils