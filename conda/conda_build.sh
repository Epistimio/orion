#!/bin/bash

set -e
set -x

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda install conda-build anaconda-client

conda build conda

if [[ -n "${TRAVIS_TAG}" ]]
then
    mkdir -p conda-bld/linux-64
    cp $HOME/miniconda/conda-bld/linux-64/orion* conda-bld/linux-64/
    conda convert --platform all conda-bld/linux-64/orion* --output-dir conda-bld/
    anaconda -t $ANACONDA_TOKEN upload conda-bld/**/orion*
fi
