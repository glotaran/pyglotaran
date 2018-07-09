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


.. _Github repo: https://github.com/glotaran/glotaran
.. _tarball: https://github.com/glotaran/glotaran/tarball/develop
