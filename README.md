# pyglotaran

<img align="right" width="100" height="100" src="https://raw.githubusercontent.com/glotaran/pyglotaran/main/docs/source/images/pyglotaran_logo_transparent.svg">

[![PyPI version](https://badge.fury.io/py/pyglotaran.svg)](https://badge.fury.io/py/pyglotaran)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyglotaran.svg)](https://anaconda.org/conda-forge/pyglotaran)
![Tests](https://github.com/glotaran/pyglotaran/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyglotaran/badge/?version=latest)](https://pyglotaran.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/gh/glotaran/pyglotaran/branch/master/graph/badge.svg)](https://codecov.io/gh/glotaran/pyglotaran)
[![CodeQL](https://github.com/glotaran/pyglotaran/actions/workflows/codeql.yml/badge.svg)](https://github.com/glotaran/pyglotaran/actions/workflows/codeql.yml)
[![Discord](https://img.shields.io/discord/883443835135475753.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/KfnEYRSTJx)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4534043.svg)](https://doi.org/10.5281/zenodo.4534043)

A framework for Global and Target Analysis written in Python.

## The Future of Global and Target Analysis

A scientific publication titled "pyglotaran: a lego-like Python framework for global and target analysis of time-resolved spectra" (DOI: [10.1007/s43630-023-00460-y](https://doi.org/10.1007/s43630-023-00460-y)) covers aspects of the architecture and the design of the software whille illustrating its flexibility as an analysis tool through some exciting case studies. This publication, along with other [pyglotaran-publications](https://github.com/glotaran/pyglotaran-publications) demonstrates why we believe this framework is the future of global and target analysis.

## Usage of pyglotaran

<sub>**Warning**: This is an _early access_ release, please refer to the [usage notice](#usage-notice) down below prior to committing to use pyglotaran to avoid surprises down the line.</sub>

A common use case for the framework is the analysis of time-resolved spectroscopy measurements in the study of energy transfer pathways in photosynthesis, or the characterization of energy transfer (in-)efficiencies in photovoltaics.

pyglotaran can be used from a Python script, or ideally Notebook, and involves specifying your desired analysis scheme consisting of a `model` and its `parameters` together with your `experiment_data`, and then letting it `optimize` this for you. This will fitting your data while optimizing for the residuals given the model you specified, the constraints you specified therein given the (free) parameters and its starting values you provided.

We have prepared a number of comprehensive examples in the form of python notebooks in the [pyglotaran-examples](https://github.com/glotaran/pyglotaran-examples) which illustrate how to use the frameowork. Download the example that best aligns with your use case, and give it go!

### Usage notice

This software package is made available as an early access release, to the advantage of the (scientific) community who wishes to make use of it before it has fully matured, but without any warranties.

Anyone using this package for serious work - scientists and academic users in particular - are cautioned, and treat it as any other instrument or tool that requires calibration or validation. Also be prepared for some refactoring of models or analysis specifications, sometime down the line.

That said, the pyglotaran package has been used in several peer-reviewed scientific publications, and it has been partially cross-validated against comparable software, such as the [R-package TIMP](https://dx.doi.org/10.18637/jss.v018.i03), and the TIM software described in [this publication](https://doi.org/10.1016/j.bbabio.2004.04.011) ([DOI: 10.1016/j.bbabio.2004.04.011](https://doi.org/10.1016/j.bbabio.2004.04.011)).

The examples used in this validation process can be obtained from the [pyglotaran-examples repository](https://github.com/glotaran/pyglotaran-examples).

Since the early access version `v0.6.0` it was used in scientific teaching by a number of students in the Photosynthesis and Energy course from 2022-2024 under the supervision of [dr. Ivo van Stokkum](https://www.nat.vu.nl/~ivo/) ([profile](https://research.vu.nl/en/persons/ihm-van-stokkum), github: [ism200](https://github.com/ism200/)). Course material can be found here: [ism200\PE2022](https://github.com/ism200/PE2022/)

## Glotaran legacy

The pyglotaran package derives its name from the Glotaran software package (now called [glotaran-legacy](https://github.com/glotaran/glotaran-legacy)), first released in 2011 and described in a highly-cited publication in the Journal of Statistical Software, under the title [Glotaran: A Java-Based Graphical User Interface for the R Package TIMP](https://www.jstatsoft.org/article/view/v049i03) ([DOI: 10.18637/jss.v049.i03](https://dx.doi.org/10.18637/jss.v049.i03)).

The [pyglotaran](https://github.com/glotaran/pyglotaran) framework can be considered the spiritual successor of the [glotaran-legacy](https://github.com/glotaran/glotaran-legacy) software and has the backing of many of its original creators.

## Community Support

For questions / suggestion please reach out to us via:

1. [GitHub issues](https://github.com/glotaran/pyglotaran/issues)
2. [Discord](https://discord.gg/KfnEYRSTJx)
3. [Google-Groups mailing list](https://groups.google.com/forum/#!forum/glotaran)

## Credits

The credits can be found in the documentation
[authors section](https://pyglotaran.readthedocs.io/en/latest/authors.html)
