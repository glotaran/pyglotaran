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
    glotaran.analysis.grouping
    glotaran.analysis.nnls
    glotaran.analysis.optimize
    glotaran.analysis.result
    glotaran.analysis.simulation
    glotaran.analysis.variable_projection
    glotaran.cli
    glotaran.cli.main
    glotaran.cli.util
    glotaran.examples
    glotaran.examples.sequential
    glotaran.io
    glotaran.io.chlorospec_format
    glotaran.io.external_file_formats
    glotaran.io.external_file_formats.sdt_file
    glotaran.io.legacy_readers
    glotaran.io.mlsd_file_format
    glotaran.io.prepare_dataset
    glotaran.io.sdt_file_reader
    glotaran.io.wavelength_time_explicit_file
    glotaran.model
    glotaran.model.dataset_descriptor
    glotaran.model.model
    glotaran.model.model_attribute
    glotaran.model.model_decorator
    glotaran.model.model_property
    glotaran.model.util
    glotaran.models
    glotaran.models.doas
    glotaran.models.doas.doas_matrix
    glotaran.models.doas.doas_megacomplex
    glotaran.models.doas.doas_model
    glotaran.models.doas.doas_result
    glotaran.models.doas.doas_spectral_matrix
    glotaran.models.doas.oscillation
    glotaran.models.flim
    glotaran.models.flim.flim_model
    glotaran.models.spectral_temporal
    glotaran.models.spectral_temporal.initial_concentration
    glotaran.models.spectral_temporal.irf
    glotaran.models.spectral_temporal.k_matrix
    glotaran.models.spectral_temporal.kinetic_matrix
    glotaran.models.spectral_temporal.kinetic_megacomplex
    glotaran.models.spectral_temporal.kinetic_model
    glotaran.models.spectral_temporal.kinetic_result
    glotaran.models.spectral_temporal.spectral_constraints
    glotaran.models.spectral_temporal.spectral_matrix
    glotaran.models.spectral_temporal.spectral_relations
    glotaran.models.spectral_temporal.spectral_shape
    glotaran.models.spectral_temporal.spectral_temporal_dataset_descriptor
    glotaran.parameter
    glotaran.parameter.parameter
    glotaran.parameter.parameter_group
    glotaran.parse
    glotaran.parse.parser
    glotaran.parse.register
