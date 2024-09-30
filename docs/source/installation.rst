.. highlight:: shell

Installation
============


Prerequisites
-------------

* Python 3.10 or higher
* (Recommended) `uv <https://docs.astral.sh/uv/>`_ installed and on the path.
* Basic familiarity with command-line interfaces.


Recommended installation method: using uv
-----------------------------------------

It appears the Python community is quickly converging to uv as the preferred way to install Python and Python packages across all platforms, so this is what we will be recommending.
They have an excellent `getting started <https://docs.astral.sh/uv/getting-started>`_ guide available, which explains how to `set it up <https://docs.astral.sh/uv/getting-started/installation>`_.

If you go down this route, note that uv can also be used to install Python itself, so you don't have to install it separately.

1. Install ``uv``: follow the getting started guide (linked above) to set it up for your platform.

2. Use ``uv`` to install Python (if not already installed):

   .. code-block:: shell

      uv python install # Automatically installs the latest Python version
      # or use `uv python install 3.10`` to install a specific Python version

3. Create a virtual environment and activate it:

   .. code-block:: shell

      uv venv
      source .venv/bin/activate  # On Unix or macOS
      .venv\Scripts\activate  # On Windows (both cmd.exe and PowerShell)

4. Install pyglotaran:

   .. code-block:: shell

      uv pip install pyglotaran pyglotaran-extras jupyterlab

.. note::

   If you use PowerShell you may have to set your LocalMachine's Execution Policy to RemoteSigned to be able to run scripts.
   To do so, start a new PowerShell session as Administrator and run the following command:

   .. code-block:: shell

      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine

   This makes sure that local scripts are not required to be signed, but still prevents running scripts from the internet.


From sources
------------

Prerequisites
~~~~~~~~~~~~~

* git installed

Follow the steps outlined above to setup a virtual environment and activate it.

.. code-block:: shell

   $ uv pip install https://github.com/glotaran/pyglotaran.git

For updating pyglotaran, just re-run the command above.

If you prefer to manually download the source files, you can find them on `Github`_. Alternatively you can clone them with `git`_ (preferred):

.. code-block:: shell

   $ git clone https://github.com/glotaran/pyglotaran.git

Within a terminal, navigate to directory where you have unpacked or cloned the code and enter

.. code-block:: shell

   $ uv pip install -e .[full]

For updating, simply download and unpack the newest version (or run ``$ git pull`` in pyglotaran directory if you used `git`_) and and re-run the command above.

.. _Github: https://github.com/glotaran/pyglotaran
.. _git: https://git-scm.com/
.. _uv_docs: https://docs.astral.sh/uv/
.. _uv_github: https://github.com/astral-sh/uv
