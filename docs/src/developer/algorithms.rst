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
independently from the core metaopt package. In fact, the core package is able
to discover externally installed algorithms implementing metaopt's interface
**as plugins**!

.. contents:: Developer's Guide 101: Basic Algorithm

***********************
Implementing Algorithms
***********************

Implement BaseAlgorithm
=======================

The Space class
===============

Distribute your Algorithms
==========================
