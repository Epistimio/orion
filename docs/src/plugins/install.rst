**********
Installing
**********

Oríon is built to be easily extensible. Algorithms and database backends can be implemented in
external repositories and installed as plugins. The installation process is very simple, you only
need to install them as you would for Oríon's core (see :doc:`/install/core`). They will be
automatically detected and available for Oríon at run-time.

For more information about how to develop your own plugins, see section :doc:`/plugins/algorithms`.

Algorithms
==========

Note that active development is currently focused on the core. Therefore only one algorithm plugin
is available for now.

Skopt algorithms
----------------

``orion.algo.skopt`` provides a wrapper for `Bayesian optimizer`_ using Gaussian process implemented
in `scikit optimize`_. For more information about algorithms configuration and usage please refer to
:doc:`/user/algorithms`.

.. _scikit optimize: https://scikit-optimize.github.io/
.. _bayesian optimizer: https://scikit-optimize.github.io/#skopt.Optimizer

.. code-block:: sh

   pip install orion.algo.skopt


Next we define the file ``bayes.yaml`` as this

.. code-block:: yaml

    experiment:
        name: orion-with-bayes
        algorithms: BayesianOptimizer

Then call ``orion hunt`` with the configuration file.

.. code-block:: bash

    $ orion hunt --config bayes.yaml ./script.sh --lr~'loguniform(1e-5, 1.0)'

Now, we have a Bayesian optimizer sampling learning-rate values to optimize the error rate.
