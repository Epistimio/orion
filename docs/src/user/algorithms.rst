.. _Setup algorithm:

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
       algorithm:
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
        algorithm:
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
        algorithm:
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
        algorithm:
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
        algorithm:
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


.. _BOHB-algorithm:

BOHB
----

`BOHB`_, is an integration of a Bayesian Optimization algorithm for the
selection of hyperparameters to try at the first rung of Hyperband brackets.
First batch of Trials will be sampled randomly, but subsequent ones will be
selected using Bayesian Optimization.
See :ref:`hyperband-algorithm` for more information on how to use multi-fidelity algorithms.

.. _BOHB: https://arxiv.org/abs/1807.01774

.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

  experiment:
    algorithm:
      bohb:
        min_points_in_model: 20
        top_n_percent: 15
        num_samples: 64
        random_fraction: 0.33
        bandwidth_factor: 3
        min_bandwidth": 1e-3
        parallel_strategy:
          of_type: StatusBasedParallelStrategy
          strategy_configs:
             broken:
                of_type: MaxParallelStrategy


.. autoclass:: orion.algo.bohb.BOHB
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration


.. _DEHB-algorithm:

DEHB
----


`DEHB`_, is an integration of a Differential Evolutionary algorithm with Hyperband.
While BOHB, uses Bayesian Optimization to select the hyperparameter to try
at the first rung of subsequent brackets, DEHB uses Differential Evolution for both
selecting the hyperparameters to try at the first rung of subsequent brackets and to mutate
best sets of hyperparameters when promoting trials inside a bracket.
Trials cannot be resumed after promotion to higher fidelity level with DEHB.
DEHB leads to different hyperparameter values and thus different trial ids, for this
reason trials cannot be resumed after promotions as for other variants of hyperbands.
See :ref:`hyperband-algorithm` for more information on how to use multi-fidelity algorithms.

.. _DEHB: https://arxiv.org/abs/2105.09821


.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

  experiment:
    algorithm:
      dehb:
        seed: null
        mutation_factor: 0.5
        crossover_prob: 0.5
        mutation_strategy: rand1
        crossover_strategy: bin
        boundary_fix_type: random
        min_clip: null
        max_clip: null
        max_age: 10e10


.. autoclass:: orion.algo.dehb.dehb.DEHB
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration

.. _PBT:

Population Based Training (PBT)
-------------------------------

.. warning::

   PBT was broken in version v0.2.4. Make sure to use the latest release.

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

    algorithm:
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


.. _PB2:

Population Based Bandits (PB2)
------------------------------

.. warning::

   PB2 was broken in version v0.2.4. Make sure to use the latest release.

Population Based Bandits is a variant of Population Based Training using probabilistic model to
guide
the search instead of relying on purely random perturbations.
PB2 implementation uses a time-varying Gaussian process to model the optimization curves
during training. This implementation is based on `ray-tune`_ implementation. Oríon's version
supports discrete and categorical dimensions, and offers better resiliency to broken
trials by using back-tracking.

.. _ray-tune: https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pb2_utils.py

See documentation below for more information on the algorithm and how to use it.

.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

  experiment:

    algorithm:
      pb2:
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



.. autoclass:: orion.algo.pbt.pb2.PB2
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
        algorithm:
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


.. _ax-algorithm:

Ax
--

`Ax`_ is a platform for optimizing any kind of experiment, including machine
learning experiments, A/B tests, and simulations. Ax can optimize discrete
configurations (e.g., variants of an A/B test) using multi-armed bandit
optimization, and continuous (e.g., integer or floating point)-valued
configurations using Bayesian optimization.

.. _ax: https://ax.dev/

Configuration
~~~~~~~~~~~~~


.. code-block:: yaml

  experiment:
    algorithm:
      ax:
        seed: 1234
        n_initial_trials: 5,
        parallel_strategy:
          of_type: StatusBasedParallelStrategy
          strategy_configs:
             broken:
                of_type: MaxParallelStrategy


.. autoclass:: orion.algo.axoptimizer.AxOptimizer
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration






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
        algorithm:
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


.. _mofa algorithm:

MOFA
-----

The MOdular FActorial Design (`MOFA`_) algorithm is based on factorial design and factorial
analysis methods to optmimize hyperparameters. It performs multiple iterations each of which
starts with sampling hyperparameter trial values from an orthogonal latin hypercube to cover
the search space well while de-correlating hyperparameters. Once all trials in an iteration
are returned, MOFA performs factorial analysis to determine which hyperparameters should be
fixed in value and which hyperparameters require further exploration. As the hyperparameters
become fixed, the number of trials are reduced in subsequent iterations.

.. _MOFA: https://arxiv.org/abs/2011.09545

.. note::

   MOFA requires Python v3.8 or greater and scipy v1.8 or greater.

.. note::

   Default values for the ``index``, ``n_levels``, and ``strength`` parameters are set
   to the empirically obtained optimal values described in section 5.2 of the paper.
   The ``strength`` parameter must be set to either ``1`` or ``2``.

.. note::

   The number of trials N for a single MOFA iteration is set to ``N = index * n_levels^strength``.
   The ``--exp-max-trials`` should at least be a multiple of ``N``.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithm:
            MOFA:
               seed: null
               index: 1
               n_levels: 5
               strength: 2
               threshold: 0.1

.. autoclass:: orion.algo.mofa.mofa.MOFA
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     requires_dist, requires_type


.. _nevergrad-algorithm:


Nevergrad
---------

`Nevergrad`_ is a derivative-free optimization platform providing
a library of algorithms for hyperparameter search.

.. _nevergrad: https://facebookresearch.github.io/nevergrad/

.. code-block:: yaml

    experiment:
        algorithm:
            nevergrad:
                seed: null
                budget: 1000
                num_workers: 10
                model_name: NGOpt


.. autoclass:: orion.algo.nevergradoptimizer.NevergradOptimizer
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     requires_dist, requires_type



.. _HEBO-algorithm:

HEBO
----

.. warning::

   HEBO package does not work with numpy>=1.24.0.


Evolutionary algorithms from the `HEBO`_ repository are made available in Orion. There are a wide
range of configutaion options for these algorithms, including the choice of model, evolutionary
strategy, and acquisition function.

.. _HEBO: https://github.com/huawei-noah/HEBO


Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   experiment:
      algorithm:
         hebo:
            seed: 1234
            parameters:
               model_name: catboost
               random_samples: 5
               acquisition_class: hebo.acquisitions.acq.MACE
               evolutionary_strategy: nsga2
               model_config: null

.. autoclass:: orion.algo.hebo.hebo_algo.HEBO
   :noindex:
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng,
                     configuration




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
