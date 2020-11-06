#!/bin/bash

./conda/conda_test.sh

if [[ -n "${TRAVIS_TAG}" ]]
then
    mkdir -p conda-bld/linux-64
    cp $HOME/miniconda/conda-bld/linux-64/orion* conda-bld/linux-64/
    conda convert --platform all conda-bld/linux-64/orion* --output-dir conda-bld/
    anaconda -t $ANACONDA_TOKEN upload conda-bld/**/orion*
fi
