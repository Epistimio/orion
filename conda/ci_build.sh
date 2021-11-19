#!/bin/bash

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda config --set channel_priority strict

pip uninstall -y setuptools
conda install -c anaconda setuptools

conda update -q conda
conda info -a
conda install conda-build anaconda-client

conda build conda --python 3.7
conda build conda --python 3.8
conda build conda --python 3.9
conda build conda --python 3.10
