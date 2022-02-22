.. _Setup Algorithms:

**********
Algorithms
**********

.. contents::
   :depth: 2
   :local:

Default algorithm is a random search based on the probability
distribution given to a search parameter's definition.

Selecting and Configuring
=========================

In a Oríon configuration YAML, define:

.. code-block:: yaml

   experiment:
       algorithms:
           gradient_descent:
               learning_rate: 0.1

In this particular example, the name of the algorithm extension class to be
imported and instantiated is ``Gradient_Descent``, so the lower-case identifier
corresponds to it.

All algorithms have default arguments that should work reasonably well in general.
To tune the algorithm for a specific problem, you can set those arguments in the
yaml file as shown above with ``learning_rate``.

Included Algorithms
===================

.. contents::
   :depth: 1
   :local:

.. _random-search:

Random Search
-------------

Random search is the most simple algorithm. It samples from given priors. That's it.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            random:
                seed: null


.. autoclass:: orion.algo.random.Random
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng


.. _grid-search:

Grid Search
-----------

Grid search is one of the simplest algorithm. It can work reasonably well for small search spaces of
one or two dimensions but should be avoided for larger search spaces. The search space can be
configured in three different ways.

1. Default for the lazy. You can set a very large ``n_values`` (ex: 100) and the grid will be
   adjusted so that it results in less than ``max_trials`` as defined in the experiment
   configuration.
2. You can set ``n_values`` to the number of points desired by dimension. Note that if this
   leads to too many trials the grid will be shrunken down to fit below ``max_trials``.
3. You can pass a dictionary to ``n_values`` specifying the number of points for each dimensions.
   Ex: ``n_values: {'dim1': 3, 'dim2': 4}``.

.. note::

   For categorical dimensions (``choices()``) all values are used to build the grid. This means
   ``n_values`` will not be honored. A warning is printed when this happens. Accordingly,
   if too many options are provided for the categorical dimensions the grid may lead to more trials
   than ``max_trials``. A ``ValueError`` will be raised in such scenario.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            gridsearch:
                n_values: 100


.. autoclass:: orion.algo.gridsearch.GridSearch
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration, requires_dist, requires_type, build_grid



.. _hyperband-algorithm:

Hyperband
---------

`Hyperband`_ extends the `SuccessiveHalving`_ algorithm by providing a way to exploit a
fixed budget with different number of configurations for ``SuccessiveHalving`` algorithm to
evaluate. Each run of ``SuccessiveHalving`` will be defined as a ``bracket`` in Hyperband.
Hyperband requires two inputs (1) ``R``, the maximum amount of resource that can be allocated
to a single configuration, and (2) ``eta``, an input that controls the proportion of
configurations discarded in each round of SuccessiveHalving.

To use Hyperband in Oríon, you must specify one parameter with ``fidelity(low, high, base)``
as the prior, ``low`` will be ignored, ``high`` will be taken as the maximum resource ``R``
and ``base`` will be taken as the reduction factor ``eta``.

Number of epochs usually can be used as the resource but the algorithm is generic and can be
applied to any multi-fidelity setting. That is, you can use training time, specifying the
fidelity with ``--epochs~fidelity(low=1, high=81, base=3)``
(assuming your script takes this argument in commandline),
but you could also use other fidelity
such as dataset size ``--dataset-size~fidelity(low=500, high=50000)``
(assuming your script takes this argument and adapt dataset size accordingly).


.. _SuccessiveHalving: https://arxiv.org/abs/1502.07943

.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            hyperband:
                seed: null
                repetitions: 1


.. autoclass:: orion.algo.hyperband.Hyperband
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration, sample_from_bracket, append_brackets, create_bracket,
                     create_brackets, promote, register_samples, sample, seed_brackets,
                     executed_times



.. _ASHA:

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

The most common way of using ASHA is to reduce the number of epochs,
but the algorithm is generic and can be applied to any multi-fidelity setting.
That is, you can use training time, specifying the fidelity with
``--epochs~fidelity(low=1, high=100)``
(assuming your script takes this argument in commandline),
but you could also use other fidelity
such as dataset size ``--dataset-size~fidelity(low=500, high=50000)``
(assuming your script takes this argument and
adapt dataset size accordingly). The placeholder ``fidelity(low, high)`` is a special prior for
multi-fidelity algorithms.


.. _asynchronous successive halving algorithm: https://arxiv.org/abs/1810.05934
.. _Hyperband: https://arxiv.org/abs/1603.06560

.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            asha:
                seed: null
                num_rungs: null
                num_brackets: 1
                repetitions: 1


.. autoclass:: orion.algo.asha.ASHA
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration, sample_from_bracket, append_brackets, create_bracket,
                     create_brackets, promote, register_samples, sample, seed_brackets,
                     executed_times, compute_bracket_idx


.. _PBT:

Population Based Training (PBT)
-------------------------------

Population based training is an evolutionary algorithm that evolve trials
from low fidelity levels to high fidelity levels (ex: number of epochs), reusing
the model's parameters along the way. This has the effect of creating hyperparameter
schedules through the fidelity levels.

See documentation below for more information on the algorithm and how to use it.

.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

  experiment:

    strategy: StubParallelStrategy

    algorithms:
      pbt:
        population_size: 50
        generations: 10
        fork_timeout: 60
        exploit:
          of_type: PipelineExploit
          exploit_configs:
            - of_type: BacktrackExploit
              min_forking_population: 5
              truncation_quantile: 0.9
              candidate_pool_ratio: 0.2
            - of_type: TruncateExploit
              min_forking_population: 5
              truncation_quantile: 0.8
              candidate_pool_ratio: 0.2
         explore:
           of_type: PipelineExplore
           explore_configs:
             - of_type: ResampleExplore
               probability: 0.2
             - of_type: PerturbExplore
               factor: 1.2
               volatility: 0.0001



.. note::
   Notice the additional ``strategy`` in configuration which is not mandatory for most other
   algorithms. See :ref:`StubParallelStrategy` for more information.


.. autoclass:: orion.algo.pbt.pbt.PBT
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration, requires_type, rng, register



.. _tpe-algorithm:

TPE
---

`Tree-structured Parzen Estimator`_ (TPE) algorithm is one of Sequential Model-Based
Global Optimization (SMBO) algorithms, which will build models to propose new points based
on the historical observed trials.

Instead of modeling p(y|x) like other SMBO algorithms, TPE models p(x|y) and p(y),
and p(x|y) is modeled by transforming that generative process, replacing the distributions of
the configuration prior with non-parametric densities.

The TPE defines p(x|y) using two such densities l(x) and g(x) where l(x) is distribution of
good points and g(x) is the distribution of bad points. Good and bad points are split from observed
points so far with a parameter `gamma` which defines the ratio of good points. New point candidates
will be sampled with l(x) and Expected Improvement (EI) optimization scheme will be used to find
the most promising point among the candidates.


.. _Tree-structured Parzen Estimator:
    https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf

.. note::

   Current implementation only supports uniform, loguniform, uniform discrete and choices as prior.
   As for choices prior, the probabilities if any given will be ignored.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            tpe:
                seed: null
                n_initial_points: 20
                n_ei_candidates: 25
                gamma: 0.25
                equal_weight: False
                prior_weight: 1.0
                full_weight_num: 25
                parallel_strategy:
                    of_type: StatusBasedParallelStrategy
                    strategy_configs:
                        broken:
                            of_type: MaxParallelStrategy
                    default_strategy:
                        of_type: NoParallelStrategy


.. autoclass:: orion.algo.tpe.TPE
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration, sample_one_dimension, split_trials, requires_type



.. _evolution-es algorithm:

Evolution-ES
-------------

`Evolution-ES`_, the evolution algorithm with early stop version.
Here is an implementation of `Evolution-ES`_.
In the evolution algorithm, we follow the tournament selection algorithm
as `Large-Scale-Evolution`_.
Tournament selection evolutionary hyper-parameter search is conducted
by first defining a gene encoding
that describes a hyper-parameter combination,
and then creating the initial population by randomly
sampling from the space of gene encodings to create individuals,
which are trained and assigned fitnesses.
The population is then repeatedly sampled from to produce groups,
and the parent is selected by the individual
with the highest fitness. Selected parents have their gene encodings
mutated to produce child models.
Individual in the group with the lowest fitness is killed,
while the newly evaluated child model is added to
the population, taking the killed individual’s place.
This process is repeated and results
in a population with high fitness individuals can
represent the good hyper-parameter combination.
`Evolution-ES`_ also formulated a method to dynamically allocate
resources to more promising individual
according to their fitness, which is referred to as
Progressive Dynamic Hurdles (PDH),
allows individuals that are consistently performing well to train for more steps.
It can be roughly interpreted as a sophisticated random search
that leverages partial information of the trial execution to concentrate resources
on the most promising ones.

The implementation follows the process and use way of Hyperband.
Additionally, The fidelity base in Evolution-ES can be
extended to support ``fidelity(low, high, base=1)``,
which is the same as ``linspace(low, high)``.

.. _Evolution-ES: https://arxiv.org/abs/1901.11117
.. _Large-Scale-Evolution: https://arxiv.org/abs/1703.01041


Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            EvolutionES:
                seed: null
                repetitions: 1
                nums_population: 20
                mutate:
                    function: orion.algo.mutate_functions.default_mutate
                    multiply_factor: 3.0
                    add_factor: 1


.. autoclass:: orion.algo.evolution_es.EvolutionES
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     requires_dist, requires_type


Algorithm Plugins
=================

Plugins documentation is hosted separately. See short documentations below to find
links to full plugins documentation.

.. _skopt-plugin:

Scikit-Optimize
---------------

This package is a plugin providing a wrapper for
`skopt <https://scikit-optimize.github.io>`__ optimizers.

For more information, you can find the documentation at
`orionalgoskopt.readthedocs.io <https://orionalgoskopt.readthedocs.io>`__.


.. _robo-plugin:

Robust Bayesian Optimization
----------------------------

This package is a plugin providing a wrapper for
`RoBO <https://github.com/automl/robo>`__ optimizers.

You will find in this plugin many models for Bayesian Optimization:
`Gaussian Process <https://epistimio.github.io/orion.algo.robo/usage.html#robo-gaussian-process>`__,
`Gaussian Process with MCMC <https://epistimio.github.io/orion.algo.robo/usage.html#robo-gaussian-process-with-mcmc>`__,
`Random Forest <https://epistimio.github.io/orion.algo.robo/usage.html#robo-random-forest>`__,
`DNGO <https://epistimio.github.io/orion.algo.robo/usage.html#robo-dngo>`__ and
`BOHAMIANN <https://epistimio.github.io/orion.algo.robo/usage.html#robo-bohamiann>`__.

For more information, you can find the documentation at
`epistimio.github.io/orion.algo.robo <https://epistimio.github.io/orion.algo.robo>`__.

.. _parallel-strategies:

Parallel Strategies
===================

A parallel strategy is a method to improve parallel optimization
for sequential algorithms. Such algorithms can only observe
trials that are completed and have a corresponding objective.
To get around this, parallel strategies produces *lies*,
noncompleted trials with fake objectives, which can
be used by algorithms to avoid exploring space nearby pending or broken trials.
The strategies will differ in how they assign objectives
to the *lies*.

NoParallelStrategy
------------------

Does not return any lie. This is useful to benchmark parallel
strategies and measure how they can help compared to no
strategy.

.. autoclass:: orion.algo.parallel_strategy.NoParallelStrategy
   :noindex:
   :exclude-members: state_dict, set_state, infer, lie, configuration, observe

.. _StubParallelStrategy:

StubParallelStrategy
--------------------

Assign to *lies* an objective of ``None`` so that
non-completed trials are observed and identifiable by algorithms
that can leverage parallel optimization.

The value of the objective is customizable with ``stub_value``.

.. autoclass:: orion.algo.parallel_strategy.StubParallelStrategy
   :noindex:
   :exclude-members: state_dict, set_state, infer, lie, configuration, observe

.. _MaxParallelStrategy:

MaxParallelStrategy
-------------------

Assigns to *lies* the best objective observed so far.

The default value assigned to objective when less than 1 trial
is completed is configurable with ``default_result``. It
is ``float('inf')`` by default.

.. autoclass:: orion.algo.parallel_strategy.MaxParallelStrategy
   :noindex:
   :exclude-members: state_dict, set_state, infer, lie, configuration, observe


.. _MeanParallelStrategy:

MeanParallelStrategy
--------------------

Assigns to *lies* the mean of all objectives observed so far.

The default value assigned to objective when less than 2 trials
are completed is configurable with ``default_result``. It
is ``float('inf')`` by default.

.. autoclass:: orion.algo.parallel_strategy.MeanParallelStrategy
   :noindex:
   :exclude-members: state_dict, set_state, infer, lie, configuration, observe

.. _StatusBasedParallelStrategy:

StatusBasedParallelStrategy
---------------------------

Uses a different strategy based on the status of the trial at hand.

.. autoclass:: orion.algo.parallel_strategy.StatusBasedParallelStrategy
   :noindex:
   :exclude-members: state_dict, set_state, infer, lie, configuration, observe, get_strategy
