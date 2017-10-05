# Asynchronous Hyperparameter Optimization on Distributed N-Node CPU/GPUs

# Requirements

- cmake 3.8.0 (minimum)
- gcc-7.2 (minimum)
- boost 1.65.1 (minimum)

# Research Plan

The plan of this package is to build an automatic tool to do hyperparameter optimization. During our time at MILA, before and after coding up research ideas we always do hyperparameter optimization to set up a baseline on SOTA.
Previous methods which are grid search and random search requires tedious work on grad student part to handpicking parameters. We realize that this process could be automated.

Another point is that we want a tool that does not necessarily work for big labs like MILA, but for labs with just 5 computers with GPUs together connected with a router or a startup company in Silicon Valley that could only
afford low amount of AWS machines/GPUs, that doesn't have one reliable central node to control node coordination, new hyperparameter choice and node pruning. We plan to use a decentralize system that use effective broadcast of
hyperparameter choice to each node, and also saving the hyperparameter history on a TinyDB/levelDB. Although we would not hash all of the hyperparameter blocks to ensure we didn't choose the same hyperparameters across two different
nodes in the network.

The choice of new hyperparameters would be done using Bayesian optimization algorithms or evolutionary strategies/genetic algorithms. We would want to implement early stopping conditions such as Freeze-Thaw.

# Execution of Research Plan

We plan to build this package that would be language agnostic, and also operation system agnostic, although at the beginning we would want to use it for mainly python scripts. This package is written mainly in C++17 and we would want to accomodate C++20. 

# Installation

`mkdir build`
`cd build`
`cmake ..`
`make -j8`
`(sudo) make install`

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

Dendi Suhubdy, Xavier Bouthillier, Christos Tsirigotis. Supervised by Pascal Lamblin and Frédéric Bastien.

# Copyright

Univeriste de Montreal, 2017.

# License

GNU GPL 
