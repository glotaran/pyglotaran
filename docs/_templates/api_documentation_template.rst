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

    {}
