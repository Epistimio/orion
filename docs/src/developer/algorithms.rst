The absolute bare **minimum interface** an optimization algorithm needs to have
is an ``observe`` method and a ``suggest`` method, so that:

 1. An algorithm can ``observe`` the **results** associated with the evaluation
    of a list of points in the parameter space. Using the history of evaluated
    attempts, the algorithm can estimate better points to evaluate.
 2. An algorithm can ``suggest`` **new points** in the parameter space to try.

An attribute given to all algorithms is an object defining the **parameter
search space**. It may be useful to the developer, or an algorithm may want
to advise it.

Finally, it is suggested to distribute algorithm implementations
independently from the core Oríon package. In fact, the core package is able
to discover externally installed algorithms implementing Oríon's interface
**as plugins**!

.. contents:: Developer's Guide 104: Basic Algorithm

******************
Meet BaseAlgorithm
******************

Basic algorithm's interface is defined in :doc:`/reference/orion.algo.base`.
The example we are going to follow is from the nested source repository used for
actual functional testing, gradient_descent_algo_.

Implement Basic Algorithms
==========================

Template `src/orion/algo/gradient_descent.py <gradient_descent_algo_code>`_
TODO

The Space Class
===============

TODO

Distribute Your Algorithms
==========================

Template `setup.py <gradient_descent_algo_setup>`_
TODO

.. _gradient_descent_algo: https://github.com/mila-udem/orion/tree/master/tests/functional/gradient_descent_algo
.. _gradient_descent_algo_code: https://github.com/mila-udem/orion/blob/master/tests/functional/gradient_descent_algo/src/orion/algo/gradient_descent.py
.. _gradient_descent_algo_setup: https://github.com/mila-udem/orion/blob/master/tests/functional/gradient_descent_algo/setup.py
