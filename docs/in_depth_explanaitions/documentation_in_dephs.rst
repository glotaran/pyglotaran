Documentation "how to" in depth
===============================

How to use Sphinx
-----------------

* `Sphinx Rest Docs <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
* `Sphinx/Rest Memo <https://rest-sphinx-memo.readthedocs.io/en/latest/index.html>`_
* `Read the Docs Sphinx Theme <https://sphinx-rtd-theme.readthedocs.io/en/latest/>`_
* `Sphinx Configuration <http://www.sphinx-doc.org/en/master/usage/configuration.html>`_
* `Text <url>`_

Generate API Documentation
--------------------------

The API Documentation will be generated automatically form the docstrings.
Those Docstrings should be formatted in the
`NumPyDoc <https://numpydoc.readthedocs.io/en/latest/example.html>`_ style.
Please make use of all available features as you see fit.

The features are:
    * Parameters
    * Returns
    * Raises
    * See Also
    * Notes
    * References
    * Examples

If you add ``packages``, ``modules``, ``classes``, ``methods``, ``attributes``,
``functions`` or ``exceptions``, please read the introduction of :ref:`api-docs-new-entry`.

.. _api-docs-new-entry:

Api Documentation Creation Helper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: docs

.. automodule:: generate_api_documentation

.. autofunction:: generate_api_documentation.traverse_package

.. autofunction:: generate_api_documentation.write_api_documentation

.. autofunction:: generate_api_documentation.write_known_packages