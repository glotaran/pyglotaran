.. highlight:: shell

Installation
============


Prerequisites
-------------

* Python 3.6 or later

Windows
+++++++

The easiest way of getting Python (and some basic tools to work with it) in Windows is to use `Anaconda <https://www.anaconda.com/>`_, which provides python.

You will need a terminal for the installation. One is provided by *Anaconda* and is called *Anaconda Console*. You can find it in the start menu.

.. note::

   If you use a Windows Shell like cmd.exe or PowerShell, you might have to prefix '$PATH_TO_ANACONDA/' to all commands (e.g. *C:/Anaconda/pip.exe* instead of *pip*)

Stable release
--------------

.. warning::

   pyglotaran is early development, so for the moment stable releases are sparse and outdated.
   We try to keep the master code stable, so please install from source for now.


This is the preferred method to install pyglotaran, as it will always install the most recent stable release.

To install pyglotaran, run this command in your terminal:

.. code-block:: console

    $ pip install pyglotaran

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io/en/stable/

.. _Python installation guide: https://docs.python-guide.org/starting/installation/

If you want to install it via conda, you can run the following command:

.. code-block:: console

    $ conda install -c conda-forge pyglotaran


From sources
------------

First you have to install or update some dependencies.

Within a terminal:

.. code-block:: console

   $ pip install -U numpy scipy Cython

Alternatively, for Anaconda users:

.. code-block:: console

   $ conda install numpy scipy Cython

Afterwards you can simply use `pip`_ to install it directly from `Github`_.

.. code-block:: console

   $ pip install git+https://github.com/glotaran/pyglotaran.git

For updating pyglotaran, just re-run the command above.

If you prefer to manually download the source files, you can find them on `Github`_. Alternatively you can clone them with `git`_ (preferred):

.. code-block:: console

   $ git clone https://github.com/glotaran/pyglotaran.git

Within a terminal, navigate to directory where you have unpacked or cloned the code and enter

.. code-block:: console

   $ pip install -e .

For updating, simply download and unpack the newest version (or run ``$ git pull`` in pyglotaran directory if you used `git`_) and and re-run the command above.

.. _Github: https://github.com/glotaran/pyglotaran
.. _git: https://git-scm.com/
