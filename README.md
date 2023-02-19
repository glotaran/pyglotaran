# pyglotaran

<img align="right" width="100" height="100" src="https://raw.githubusercontent.com/glotaran/pyglotaran/main/docs/source/images/pyglotaran_logo_transparent.svg">

pyglotaran is a Python library for Global and Target Analysis

A common use case for the library is the analysis of time-resolved spectroscopy measurements in the study of energy transfer pathways in photosynthesis, or the characterization of energy transfer (in-)efficiencies in photovoltaics.

[![PyPI version](https://badge.fury.io/py/pyglotaran.svg)](https://badge.fury.io/py/pyglotaran)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyglotaran.svg)](https://anaconda.org/conda-forge/pyglotaran)
![Tests](https://github.com/glotaran/pyglotaran/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyglotaran/badge/?version=latest)](https://pyglotaran.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/glotaran/pyglotaran.git/main?urlpath=lab%2Ftree%2Fdocs%2Fsource%2Fnotebooks)\
[![Coverage Status](https://codecov.io/gh/glotaran/pyglotaran/branch/master/graph/badge.svg)](https://codecov.io/gh/glotaran/pyglotaran)
[![CodeQL](https://github.com/glotaran/pyglotaran/actions/workflows/codeql.yml/badge.svg)](https://github.com/glotaran/pyglotaran/actions/workflows/codeql.yml)
[![Discord](https://img.shields.io/discord/883443835135475753.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/KfnEYRSTJx)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4534043.svg)](https://doi.org/10.5281/zenodo.4534043)

**Warning**: This is an _early access_ release, please refer to the [usage notice](#usage-notice) down below prior to committing to use pyglotaran to avoid surprises down the line.

## Installation

Prerequisites:

- Python 3.10
- On Windows only 64bit is supported

Note for Windows Users: The easiest way to get python for Windows is via [Anaconda](https://www.anaconda.com/)

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

## Usage notice

This software package is made available as an early access release, to the advantage of the (scientific) community who wishes to make use of it before it has fully matured, but without any warranties.

Anyone using this package for serious work - scientists and academic users in particular - are cautioned, and treat it as any other instrument or tool that requires calibration or validation. Also be prepared for some refactoring of models or analysis specifications, sometime down the line.

As of yet, the pyglotaran package has not yet been used in any peer-reviewed scientific publications (contribution welcome), but it has been partially cross-validated against comparable software, such as the [R-package TIMP](https://dx.doi.org/10.18637/jss.v018.i03), and the TIM software described in [this publication](https://doi.org/10.1016/j.bbabio.2004.04.011) ([DOI: 10.1016/j.bbabio.2004.04.011](https://doi.org/10.1016/j.bbabio.2004.04.011)).

The examples used in this validation process can be obtained from the [pyglotaran-examples repository](https://github.com/glotaran/pyglotaran-examples).

An early access version of the v0.6.0 release was used in scientific teaching by a number of students in the 2022 Photosynthesis and Energy course under supervision by [dr. Ivo van Stokkum](https://www.nat.vu.nl/~ivo/) ([profile](https://research.vu.nl/en/persons/ihm-van-stokkum), github: [ism200](https://github.com/ism200/)). Course material can be found here: [ism200\PE2022](https://github.com/ism200/PE2022/)

## Glotaran legacy

The pyglotaran package derives its name from the Glotaran software package (now called [glotaran-legacy](https://github.com/glotaran/glotaran-legacy)), first released in 2011 and described in a highly-cited publication in the Journal of Statistical Software, under the title [Glotaran: A Java-Based Graphical User Interface for the R Package TIMP](https://www.jstatsoft.org/article/view/v049i03) ([DOI: 10.18637/jss.v049.i03](https://dx.doi.org/10.18637/jss.v049.i03)).

The [pyglotaran](https://github.com/glotaran/pyglotaran) software can be considered the spiritual successor of the [glotaran-legacy](https://github.com/glotaran/glotaran-legacy) software and has the backing of many of its original creators.

## The future of global and target analysis

Eventually, and hopefully sooner than later, a (scientific) publication about the architecture and the design of the software package will appear, detailing the flexibility of the software and showing why we believe this software package is the future of global and target analysis.

Until then, enjoy your glimpse into the future.

## Community Support

For questions / suggestion please reach out to us via:

1. [GitHub issues](https://github.com/glotaran/pyglotaran/issues)
2. [Discord](https://discord.gg/KfnEYRSTJx)
3. [Google-Groups mailing list](https://groups.google.com/forum/#!forum/glotaran)

## Credits

The credits can be found in the documentation
[authors section](https://pyglotaran.readthedocs.io/en/latest/authors.html)
