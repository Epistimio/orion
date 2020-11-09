#!/bin/bash

export PATH="$HOME/miniconda/bin:$PATH"

mkdir -p conda-bld/linux-64
cp $HOME/miniconda/conda-bld/linux-64/orion* conda-bld/linux-64/
conda convert --platform all conda-bld/linux-64/orion* --output-dir conda-bld/
anaconda -t $ANACONDA_TOKEN upload conda-bld/**/orion*
