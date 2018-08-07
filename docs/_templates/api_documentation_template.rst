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

    {}