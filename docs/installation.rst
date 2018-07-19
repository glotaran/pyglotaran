.. highlight:: shell

============
Installation
============

..
    Stable release
    --------------

    To install glotaran, run this command in your terminal:

    .. code-block:: console

        $ pip install glotaran

    This is the preferred method to install glotaran, as it will always install the most recent stable release.

    If you don't have `pip`_ installed, this `Python installation guide`_ can guide
    you through the process.

.. _pip: https://pip.pypa.io/en/stable/

..
    .. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for glotaran can be downloaded from the `Github repo`_.

First you have to install the Setup dependencies:

.. code-block:: console

    $ pip install -U numpy scipy Cython

Afterwards you can simply use `pip`_ to install it directly from the `Github repo`_.

.. code-block:: console

    $ pip install git+https://github.com/glotaran/glotaran.git@develop --process-dependency-links

Or you can either clone the public repository:

.. code-block:: console

    $ git clone -b develop --single-branch git://github.com/glotaran/glotaran.git

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/glotaran/glotaran/tarball/develop

And once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

Temporary workaround for Windows
--------------------------------

Since ``glotaran`` uses accelerator modules written in `Cython`_ for better peromance,
it might not be possible for you to install it, if you are missing the needed C compiler.

Which is why we created a temporary workaround for you to still enjoy the latest version of ``glotaran``,
until our build and deployment system is up and running.
Depending on your Python architecture (32bit/64bit) you can download ``glotaran for Python 3.6`` at
`Wheel for Windows 64bit`_ / `Wheel for Windows 32bit`_.

After you downloaded it you can simply install it with:

* 64bit::

    $pip install glotaran-0.0.1-cp36-cp36m-win_amd64.whl

* 32bit::

    $pip install glotaran-0.0.1-cp36-cp36m-win32.whl


.. note::  For the latest version of ``glotaran`` to run properly you should also update ``lmfit-varpro``.
           To ensure that you have the latest version of ``lmfit-varpro`` run the following command::

               $pip install --upgrade --force-reinstall git+https://github.com/glotaran/lmfit-varpro.git

.. _Cython: http://cython.org/
.. _Wheel for Windows 64bit: https://ci.appveyor.com/project/jsnel/glotaran/branch/develop/artifacts/dist%2Fglotaran-0.0.1-cp36-cp36m-win_amd64.whl
.. _Wheel for Windows 32bit: https://ci.appveyor.com/project/jsnel/glotaran/branch/develop/artifacts/dist%2Fglotaran-0.0.1-cp36-cp36m-win32.whl
.. _Github repo: https://github.com/glotaran/glotaran
.. _tarball: https://github.com/glotaran/glotaran/tarball/develop
