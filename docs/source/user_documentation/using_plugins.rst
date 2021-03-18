Plugins
=======

To be as flexible as possible ``pyglotaran`` uses a plugin system to handle new ``Models``, ``DataIo`` and ``ProjectIo``.
Those plugins can be defined by ``pyglotaran`` itself, the user or a 3rd party plugin package.

.. TODO: Write IO guide

Builtin plugin
--------------

Models
^^^^^^

- :class:`KineticSpectrumModel`
- :class:`KineticImageModel`

.. TODO: Autogenerate support tables and use them for Io Plugin instead of bullet lists.

Data Io
^^^^^^^

Plugins reading and writing data to and from :xarraydoc:`Dataset` or :xarraydoc:`DataArray`.

- :class:`AsciiDataIo`
- :class:`NetCDFDataIo`
- :class:`SdtDataIo`


Project Io
^^^^^^^^^^

Plugins reading and writing, :class:`Model`,:class:`Schema`,:class:`ParameterGroup` and :class:`Result`.

- :class:`YmlProjectIo`
- :class:`YmlProjectIo`


3rd party plugins
-----------------

Plugins not part of ``pyglotaran`` itself.

- Not yet, why not be the first? Tell us about your plugin and we will feature it here.
