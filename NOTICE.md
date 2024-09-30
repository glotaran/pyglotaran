# Usage notice for scientists

## Disclaimer

This software package is made available as an early access release, to the advantage of the (scientific) community who wishes to make use of it before it has fully matured, but without any warranties.

Anyone using this package for serious work - scientists and academic users in particular - are cautioned, and treat it as any other instrument or tool that requires calibration or validation. Also be prepared for some refactoring of models or analysis specifications, sometime down the line.

## Scientific usage

That said, the pyglotaran package has been used in several peer-reviewed scientific publications, and it has been partially cross-validated against comparable software, such as the [R-package TIMP](https://dx.doi.org/10.18637/jss.v018.i03), and the TIM software described in [this publication](https://doi.org/10.1016/j.bbabio.2004.04.011) ([DOI: 10.1016/j.bbabio.2004.04.011](https://doi.org/10.1016/j.bbabio.2004.04.011)).

The examples used in this validation process can be obtained from the [pyglotaran-examples repository](https://github.com/glotaran/pyglotaran-examples).

Since the early access version `v0.6.0` it was used in scientific teaching by a number of students in the Photosynthesis and Energy course from 2022-2024 under the supervision of [dr. Ivo van Stokkum](https://www.nat.vu.nl/~ivo/) ([profile](https://research.vu.nl/en/persons/ihm-van-stokkum), github: [ism200](https://github.com/ism200/)). Course material can be found here: [ism200\PE2022](https://github.com/ism200/PE2022/)

## Quality Control

<!-- placeholder text -->

As pyglotaran developers we strive to deliver high quality but also very reliable software, software you can trust as the basis for your scientific publications. How do we do this?

The development process follows best practices in software engineering, we use git, automated testing, continuous integration, and code reviews. All code changes must pass a comprehensive (unit) test suite and is reviewed by at least one core maintainers before being merged. We use GitHub Actions to automatically run tests, linting, and other quality checks on every pull request and code push.

But more importantly for quality assurance is the pyglotaran-validation framework. As described in Sebastian Weigand's MSc thesis, this framework allows for automated validation of pyglotaran's results against established known good results. It decouples result validation from the main project, allowing us to easily compare results from earlier with newer versions. The validation framework compares pyglotaran outputs to manually validated reference results (or so-called gold standard).

<!-- end of placeholder text -->
