****************
Setup Algorithms
****************

Default algorithm is a random search based on the probability
distribution given to a search parameter's definition.

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

All algorithms have default arguments that should work reasonably in general.
To tune the algorithm for a specific problem, use can set those arguments in the
yaml file as shown above with ``learning_rate``.

Included Algorithms
===================

Random Search
-------------

Random search is the most simple algorithm. It samples from given priors. That's it.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

     algorithms:
        random:
           seed: null


``seed``

Seed for the random number generator used to sample new trials. Default is ``None``.

ASHA
----

`Asynchronous Successive Halving Algorithm`_, the asynchronous version of
`Hyperband`_, can be roughly interpreted as a sophisticated random search that leverages
partial information of the trial execution to concentrate resources on the
most promising ones.

The main idea of the algorithm is the following. Given a fidelity dimension, such as
the number of epochs to train or the size of the dataset, ASHA samples trials
with low-fidelity and promotes the most promising ones to the next fidelity level.
This makes it possible to only execute one trial with full fidelity, leading
to very optimal resource usage.

The most common way of using ASHA is to reduce number of epochs,
but the algorithm is generic and can be applied to any multi-fidelity setting.
That is, you can use training time, specifying the fidelity with ``--epochs~fidelity()``
(assuming your script takes this argument in commandline), but you could also use other fidelity
such as dataset size ``--dataset-size~fidelity()`` (assuming your script takes this argument and
adapt dataset size accordingly). The placeholder ``fidelity()`` is a special prior for
multi-fidelity algorithms.


.. _asynchronous successive halving algorithm: https://arxiv.org/abs/1810.05934
.. _Hyperband: https://arxiv.org/abs/1603.06560

.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

     algorithms:
        asha:
           seed: null
           max_resources: 100
           grace_period: 1
           reduction_factor: 4
           num_brackets: 1

``seed``

Seed for the random number generator used to sample new trials. Default is ``None``.


``max_resources``

Maximum amount of resource that will be assigned to trials by ASHA. Only the best
performing trial will be assigned the maximum amount of resources. Default is 100.

``grace_period``

The minimum number of resource assigned to each trial. Default is 1.

``reduction_factor``

The factor by which ASHA promotes trials. If the reduction factor is 4, it means
the number of trials from one fidelity level to the next one is roughly divided by 4, and
each fidelity level has 4 times more resources than the prior one. Default is 4.

``num_brackets``

Using a grace period that is too small may bias ASHA too strongly towards fast
converging trials that do not lead to best results at convergence (stragglers).
To overcome this, you can increase the number of brackets, which increases the amount of resources
required for optimisation but decreases the bias towards stragglers. Default is 1.


Algorithm Plugins
=================

Bayesian Optimizer
------------------

``orion.algo.skopt`` provides a wrapper for `Bayesian optimizer`_ using Gaussian process implemented
in `scikit optimize`_.

.. _scikit optimize: https://scikit-optimize.github.io/
.. _bayesian optimizer: https://scikit-optimize.github.io/#skopt.Optimizer

Installation
~~~~~~~~~~~~

.. code-block:: sh

   pip install orion.algo.skopt

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

     algorithms:
        bayesopt:
           seed: null
           strategy: cl_min
           n_initial_points: 10
           acq_func: gp_hedge
           alpha: 1e-10
           n_restarts_optimizer: 0
           noise: "gaussian"
           normalize_y: False

``seed``


``strategy``

Method to use to sample multiple points.
Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.
Check skopt docs for details.

``n_initial_points``

Number of evaluations of ``func`` with initialization points
before approximating it with ``base_estimator``. Points provided as
``x0`` count as initialization points. If len(x0) < n_initial_points
additional points are sampled at random.

``acq_func``

Function to minimize over the posterior distribution. Can be:
``["LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"]``. Check skopt
docs for details.

``alpha``

Value added to the diagonal of the kernel matrix during fitting.
Larger values correspond to increased noise level in the observations
and reduce potential numerical issue during fitting. If an array is
passed, it must have the same number of entries as the data used for
fitting and is used as datapoint-dependent noise level. Note that this
is equivalent to adding a WhiteKernel with c=alpha. Allowing to specify
the noise level directly as a parameter is mainly for convenience and
for consistency with Ridge.

``n_restarts_optimizer``

The number of restarts of the optimizer for finding the kernel's
parameters which maximize the log-marginal likelihood. The first run
of the optimizer is performed from the kernel's initial parameters,
the remaining ones (if any) from thetas sampled log-uniform randomly
from the space of allowed theta-values. If greater than 0, all bounds
must be finite. Note that n_restarts_optimizer == 0 implies that one
run is performed.

``noise``

If set to "gaussian", then it is assumed that y is a noisy estimate of f(x) where the
noise is gaussian.

``normalize_y``

Whether the target values y are normalized, i.e., the mean of the
observed target values become zero. This parameter should be set to
True if the target values' mean is expected to differ considerable from
zero. When enabled, the normalization effectively modifies the GP's
prior based on the data, which contradicts the likelihood principle;
normalization is thus disabled per default.
