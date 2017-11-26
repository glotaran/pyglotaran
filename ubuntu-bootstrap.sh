#!/usr/bin/env bash

set -e 

sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get install git-core -y
git config --global color.ui true

# Install deps
sudo apt-get install python3 libfreetype6-dev libpng12-dev python-setuptools python3-dev -y
sudo apt-get install curl -y
sudo apt-get install tmux -y
sudo apt-get install build-essential gcc make g++-y

cd ~
wget https://bootstrap.pypa.io/get-pip.py
sudo -H python3 get-pip.py
python3 --version 
pip3 --version

# sudo setfacl -m user:1000:rwx /usr/local/lib/python3.5/dist-packages
# cython3 python3-scipy libfreetype6-dev -y
# sudo pip3 install sympy
# sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose -y
# sudo apt-get install python3-numpy python3-scipy python3-matplotlib ipython ipython3-notebook python3-pandas python3-nose -y

sudo -H  pip3 install numpy 
sudo -H  pip3 install scipy
sudo -H  pip3 install lmfit 
sudo -H  pip3 install jupyter
sudo -H  pip3 install matplotlib
sudo -H  pip3 install Cython
# sudo setfacl -m user:1000:rwx /usr/local/bin
sudo -H  pip3 install natsort
sudo -H  pip3 install six

# Install lmfit-varpro (not yet on PyPI - the Python Package Index)
pip3 install --user git+https://github.com/glotaran/lmfit-varpro.git

# Workaround for Bug
# sudo ln -s /usr/include/freetype2/ft2build.h /usr/include/ft2build.h

# Install glotaran
cd /vagrant/
python3 setup.py install --user

# Intall minimal latex environment to allow print to pdf in jupyter
sudo apt-get install --no-install-recommends texlive-latex-base texlive-latex-extra texlive-fonts-recommended cm-super -y
sudo apt-get install --no-install-recommends pandoc -y

# jupyter notebook --notebook-dir=/vagrant/tests/notebooks --no-browser --ip=127.0.0.1 &

# clean-up
sudo setfacl -m user:1000:r-x /usr/local/bin
rm ~/get-pip.py

