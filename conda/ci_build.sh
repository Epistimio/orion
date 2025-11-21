#!/bin/bash
PS4='[command at line $LINENO] '  # set prefix for printed commands
set -x  # make sure commands are printed before execution

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda config --add channels mila-iqia
conda config --set channel_priority strict

# pip uninstall -y setuptools  # do not uninstall-then-reinstall setuptools anymore
# conda install -c anaconda setuptools
conda install conda-build

conda update -q conda
conda info -a
conda install conda-build anaconda-client

# conda-build conda --python 3.8 falcon >=4,<5 cannot be installed on python 3.8

# So, currently, conda can build package only for python 3.9
conda-build conda --python 3.9

# Conda 3.10 does not work because of a bug inside conda itself
# Also, falcon-cors (in channel `mila-iqia`) cannot be installed on python >= 3.10
# conda build conda --python 3.10
# conda build conda --python 3.11
