# glotaran-core-python
The Python implementation of the Glotaran core

## Installation

Requirements:

* NumPy
* Cython

In folder, run:

    python3 setup.py install

## Vagrant

The repository contains a Vagrantfile which sets up a [Vagrant](https://www.vagrantup.com/) box with included Jupyter Notebook.

After installing [Vagrant](https://www.vagrantup.com/) simply go to the
repository folder and issue

    vagrant up
    # OR

    vagrant up --provider virtualbox

Note: You will need [VirtualBox](https://www.virtualbox.org/) installed.

After running `vagrant up`, open a browser and browse to `localhost:8888` (`127.0.0.1:8888`).

To shut down the box, issue

    vagrant down / halt

To connect (via ssh) and aquire a terminal on the box issue:

    vagrant ssh

To delete it (and remove all traces from your computer)

    vagrant destroy
