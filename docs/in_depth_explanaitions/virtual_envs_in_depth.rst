.. highlight:: shell

.. _virtual-envs-in-depth:

Virtual envs "how to" in depth
==============================

This Section explains how to get you virtual env up an running with different virtual env providers.

Using conda
-----------

`The full conda documentation <https://conda.io/docs/>`_.

.. note::  This is the recommended way if you use Windows.

Installation
^^^^^^^^^^^^

First you have to download
`anaconda <https://www.anaconda.com/download/>`_ (conda installation with "full" science stack)
or
`miniconda <https://conda.io/miniconda.html>`_ (minimal conda installation)
for your OS and follow its install instructions.

After that is done (maybe a restart of the terminal or PC is needed) have the ``conda`` command
available in your terminal::

    $conda update conda

Environment creation
^^^^^^^^^^^^^^^^^^^^

If that is working, create an environment::

    $conda create --name glotaran python=3.6 -y

.. note::  Python 3.7 could also be used, but packages can't be installed with ``conda install packages``
           right now. If the packages are are on PIPY already they can still be installed with
           ``pip install package``.


De-/Activating an Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To activate the environment run::

    $source activate glotaran

Or to deactivate respectively::

    $source deactivate

.. note::  On default Windows terminal (cmd/PS) you might need omit ``source`` and run
           ``activate glotaran``/``deactivate`` instead.

Using mkvirtualenv
------------------

`The full virtualenvwrapper documentation <https://virtualenvwrapper.readthedocs.io/en/latest/>`_.

Installation
^^^^^^^^^^^^

To install ``virtualenvwrapper`` run::

    $pip install virtualenvwrapper
    $source /usr/local/bin/virtualenvwrapper.sh

.. note::  Depending on your python installation you will have to search for the location of
           ``virtualenvwrapper.sh`` and change the path accordingly.

.. warning::  The line ``source /usr/local/bin/virtualenvwrapper.sh`` is for Posix Terminals and
              might not work on Windows terminals.

Environment creation
^^^^^^^^^^^^^^^^^^^^

To create an environment with ``virtualenvwrapper`` run::

    $mkvirtualenv glotaran


You should now already be in that environment::

    (glotaran)$


De-/Activating an Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To change in an existing environment from a fresh terminal run::

    $workon glotaran

Or to deactivate respectively::

    $deactivate


setting up glotaran
-------------------

Once you got your environment running you can::

    (glotaran)$git clone https://github.com/<your_name_here>/glotaran.git
    (glotaran)$cd glotaran
    (glotaran)$pip install -r requirements_dev.txt
    (glotaran)$pip install -e .
