## Installation

Prerequisites:

- Python 3.10
- On Windows only 64bit is supported

Note for Windows Users: The easiest way to get python for Windows is via the Microsoft Store

### Stable Release

To install pyglotaran from [pypi](https://pypi.org/), run this command in your terminal:

```console
pip install pyglotaran
```

If you want to install it via conda, you can run the following command:

```console
conda install -c conda-forge pyglotaran
```

To install pyglotaran together with [pyglotaran-extras](https://github.com/glotaran/pyglotaran-extras) which provides common plotting functionality you can run:

```console
pip install pyglotaran[extras]
```

### From Source

To install from source, e.g. for testing or development purposes, run these commands in your shell/terminal:

```console
git clone https://github.com/glotaran/pyglotaran.git
cd pyglotaran
pip install .
```

_**Note** (for Linux users): use pip3 instead of pip if Python2 is the system default Python installation._

_**Note** (for Anaconda users): please make sure to update your distribution prior to install since some packages managed by Anaconda cannot be updated by `pip`._
