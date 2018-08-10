***********************
Installation of plugins
***********************

Oríon is built to be easily extensible. Algorithms and database backends can be implemented in
external repositories and installed as plugins. The installation process is very simple, you only
need to install them as you would for Oríon's core (see :doc:`/install/core`). They will be
automatically detected and available for Oríon at run-time.

Note that active development is currently focused on the core. Therefore only one algorithm plugin
is available for now.

For more information about how to develop your own plugins, see section :doc:`/plugins/base`.

Algorithms
==========

Skopt algorithms
----------------

`orion.algo.skopt` provides a wrapper for `Bayesian optimizer`_ using Gaussian process implemented
in `scikit optimize`_. For more information about algorithms configuration and usage please refer to
:doc:`/user/algorithms`.

.. _scikit optimize: https://scikit-optimize.github.io/
.. _bayesian optimizer: https://scikit-optimize.github.io/#skopt.Optimizer

.. code-block:: sh

   pip install orion.algo.skopt

Database backend
================

TODO
