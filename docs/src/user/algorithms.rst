.. contents:: User's Guide 103: Algorithms


****************
Setup Algorithms
****************

Default algorithm is a random search based on the probability
distribution given to a search parameter's definition.

In the examples given below actual functional tests will be demonstrated,
in `demo <https://github.com/epistimio/orion/tree/master/tests/functional/demo>`_.

Selecting and Configuring
=========================

In a Oríon configuration YAML, define:

.. code-block:: yaml

   algorithms:
     gradient_descent:
       learning_rate: 0.1

In this particular example, the name of the algorithm extension class to be
imported and instantiated is ``Gradient_Descent``, so the lower-case identifier
corresponds to it.

Also, notice that we can configure the optimization algorithms that we would
like to use by tweaking the value of a keyword argument which is expected to
be accepted at algorithm's initialization.

Most algorithms have most of their arguments be optional and keyword. In case
that an explicit value setup is provided in the configuration file in use, then
the default values of corresponding missing values will be used; those that have
been decided by the developers of an algorithm's implementation.

List of Available Implementations
=================================

TODO

Installing Algorithms as Plugins
================================

TODO
