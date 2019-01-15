..
    Don't change api_documentation.rst since it changes will be overwritten.
    If you want to change api_documentation.rst you have to make the changes in
    api_documentation_template.rst and run `make api_docs` afterwards.
    For changes to take effect you might also have to run `make clean_all`
    afterwards.

API Documentation
=================

.. currentmodule:: glotaran

Analysis
--------

.. autosummary::
    :toctree: api

    glotaran.analysis.fitresult.FitResult

Model
-----

.. autosummary::
    :toctree: api

    glotaran.model.BaseModel
    glotaran.model.Parameter
    glotaran.model.ParameterGroup

Models
------

Kinetic Model
+++++++++++++

.. autosummary::
   :toctree: api

   glotaran.models.spectral_temporal.KineticModel
   glotaran.models.spectral_temporal.KineticMegacomplex


.. autosummary::
    :toctree: api

    glotaran.analysis
    glotaran.analysis.fitresult
    glotaran.analysis.grouping
    glotaran.analysis.nnls
    glotaran.analysis.simulation
    glotaran.analysis.variable_projection
    glotaran.data
    glotaran.data.dataio
    glotaran.data.dataio.chlorospec_format
    glotaran.data.dataio.file_readers
    glotaran.data.dataio.legacy_readers
    glotaran.data.dataio.mlsd_file_format
    glotaran.data.dataio.spectral_timetrace
    glotaran.data.dataio.wavelength_time_explicit_file
    glotaran.data.datasets
    glotaran.data.datasets.flim_dataset
    glotaran.data.datasets.high_dimensional_dataset
    glotaran.data.datasets.spectral_temporal_dataset
    glotaran.data.external_file_readers
    glotaran.data.external_file_readers.sdt_reader
    glotaran.model
    glotaran.model.base_model
    glotaran.model.dataset_descriptor
    glotaran.model.model
    glotaran.model.model_attribute
    glotaran.model.model_item
    glotaran.model.model_item_validator
    glotaran.model.parameter
    glotaran.model.parameter_group
    glotaran.models
    glotaran.models.doas
    glotaran.models.doas.doas_fit_result
    glotaran.models.doas.doas_matrix
    glotaran.models.doas.doas_megacomplex
    glotaran.models.doas.doas_model
    glotaran.models.doas.doas_spectral_matrix
    glotaran.models.doas.oscillation
    glotaran.models.flim
    glotaran.models.flim.flim_model
    glotaran.models.spectral_temporal
    glotaran.models.spectral_temporal.initial_concentration
    glotaran.models.spectral_temporal.irf
    glotaran.models.spectral_temporal.k_matrix
    glotaran.models.spectral_temporal.kinetic_fit_result
    glotaran.models.spectral_temporal.kinetic_matrix
    glotaran.models.spectral_temporal.kinetic_megacomplex
    glotaran.models.spectral_temporal.kinetic_model
    glotaran.models.spectral_temporal.kinetic_result
    glotaran.models.spectral_temporal.spectral_constraints
    glotaran.models.spectral_temporal.spectral_matrix
    glotaran.models.spectral_temporal.spectral_relations
    glotaran.models.spectral_temporal.spectral_shape
    glotaran.models.spectral_temporal.spectral_temporal_dataset_descriptor
    glotaran.parse
    glotaran.parse.parser
    glotaran.parse.register
    glotaran.plotting
    glotaran.plotting.basic_plots
    glotaran.plotting.glotaran_color_codes
