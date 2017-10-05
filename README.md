[![Build Status](https://travis-ci.org/dendisuhubdy/metaopt.svg?branch=master)](https://travis-ci.org/dendisuhubdy/metaopt)

# Asynchronous Hyperparameter Optimization on Distributed N-Node CPU/GPUs

# Requirements

- cmake 3.8.0 (minimum)
- gcc-7.2 (minimum)
- boost 1.65.1 (minimum)

# Installation

```bash
mkdir build
cd build
cmake..
make -j8
(sudo) make install
```

This will install `metaoptd` and `metaopt` binary in your computer. You will then need to install the `metaopt` Python package by

`python setup.py install`

# Usage

Run the daemon by executing

`metaoptd`

assuming that `metaoptd` is in your `$PATH`, you can also set up a cron-job to automatically start `metaoptd` when your computer (node) starts.

`metaopt` is the client that interacts with `metaoptd` (daemon service) that would submit job queues for the current node, `metaoptd` would then schedule efficiently the job based on the training statistics that we've collected frominside the experiemnt script `<python-script>`

Run your experiment by calling

`metaopt <python-script> --hyperparameter-argv`

This will spawn python as a child process and with your argument parser, we highly suggest that your hyperparameters are stated in the argument parser at the main script of your experiment. The stream of logs of the training statistics would be saved in the hyperparameter history on LevelDB/TinyDB or for a cluster/supercomputer in a MongoDB database, you would control the condition of your experiment via the python package `metaopt`

# Team

The MILA MetaOpt team (currently):

Dendi Suhubdy, Xavier Bouthillier, Christos Tsirigotis. Supervised by Pascal Lamblin and Frédéric Bastien.

# Copyright

Montreal Institute of Learning Algorithms, Univeriste de Montreal, 2017.

# License

GNU General Public License v3.0
