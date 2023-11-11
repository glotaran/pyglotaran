```{highlight} shell

```

# Installation

## Prerequisites

- Python 3.10 or 3.11

### Windows

The easiest way of getting Python (and some basic tools to work with it) in Windows is to use [Anaconda](https://www.anaconda.com/), which provides python.

You will need a terminal for the installation. One is provided by _Anaconda_ and is called _Anaconda Console_. You can find it in the start menu.

:::{note}
If you use a Windows Shell like cmd.exe or PowerShell, you might have to prefix '\$PATH_TO_ANACONDA/' to all commands (e.g. _C:/Anaconda/pip.exe_ instead of _pip_)
:::

## Stable release

:::{warning}
pyglotaran is early development, so for the moment stable releases are sparse and outdated.
We try to keep the master code stable, so please install from source for now.
:::

This is the preferred method to install pyglotaran, as it will always install the most recent stable release.

To install pyglotaran, run this command in your terminal:

```console
$ pip install pyglotaran
```

If you don't have [pip] installed, this [Python installation guide] can guide
you through the process.

If you want to install it via conda, you can run the following command:

```console
$ conda install -c conda-forge pyglotaran
```

## From sources

You can simply use [pip] to install it directly from [Github].

```console
$ pip install git+https://github.com/glotaran/pyglotaran.git
```

For updating pyglotaran, just re-run the command above.

If you prefer to manually download the source files, you can find them on [Github]. Alternatively you can clone them with [git] (preferred):

```console
$ git clone https://github.com/glotaran/pyglotaran.git
```

Within a terminal, navigate to directory where you have unpacked or cloned the code and enter

```console
$ pip install -e .
```

For updating, simply download and unpack the newest version (or run `$ git pull` in pyglotaran directory if you used [git]) and and re-run the command above.

[git]: https://git-scm.com/
[github]: https://github.com/glotaran/pyglotaran
[pip]: https://pip.pypa.io/en/stable/
[python installation guide]: https://docs.python-guide.org/starting/installation/
