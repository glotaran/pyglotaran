# glotaran-placeholder
A placeholder for the glotaran python package to be updated in the not-so-distant future.

## Instructions for use
Please do not try to install or use this package, it isn't meant to do anything yet.
If you accidentally installed it, now is the time to uninstall it, and wait for the 1.0 or 2.0 release.
=======
[![latest release](https://pypip.in/version/glotaran/badge.svg)](https://pypi.org/project/glotaran/)
[![Build Status](https://travis-ci.org/glotaran/glotaran.svg?branch=develop)](https://travis-ci.org/glotaran/glotaran)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/glotaran/glotaran?branch=develop&svg=true)](https://ci.appveyor.com/project/jsnel/glotaran?branch=develop)
[![Documentation Status](https://readthedocs.org/projects/glotaran/badge/?version=latest)](https://glotaran.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/glotaran/glotaran/badge.svg?branch=develop)](https://coveralls.io/github/glotaran/glotaran?branch=develop)

---

# Glotaran
Global and target analysis software package based on Python

## Warning
This project is still in its pre-alpha phase and undergoing rapid development, including changes to the core API, thus it is not production ready.

Proceed at your own risk!

## Additional warning for scientists
The algorithms provided by this package still need to be validated and reviewed, pending the official release it should not be used in scientific publications.

## Installation

To install glotaran, run this command in your terminal:

    pip install glotaran

This is the preferred method to install glotaran, as it will always install the most recent stable release.

## Vagrant

The repository contains a Vagrantfile which sets up a [Vagrant](https://www.vagrantup.com/) box with included Jupyter Lab.

After installing [Vagrant](https://www.vagrantup.com/) simply go to the
repository folder and issue

    vagrant up
    # OR

    vagrant up --provider virtualbox

Note: You will need [VirtualBox](https://www.virtualbox.org/) installed.

After running `vagrant up`, open a browser and browse to `localhost:9999` (`127.0.0.1:9999`) or simply doubleclick `open_vagrant-jupyter.html`.

To shut down the box, issue

    vagrant down / halt

To connect (via ssh) and aquire a terminal on the box issue:

    vagrant ssh

To delete it (and remove all traces from your computer)

    vagrant destroy

To update the glotaran-core installation on a vagrant box (without rebuilding it):

    vagrant ssh
    cd /vagrant
    sudo python3 setup.py install
    exit
    vagrant reload


## Credits

The credits can be found in the documentations
[credits section](https://glotaran.readthedocs.io/en/latest/credits.html)
