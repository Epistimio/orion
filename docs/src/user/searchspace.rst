.. _search-space:

************
Search Space
************

The search space is defined by the name of the hyperparameters to optimize and their corresponding
distribution priors. These priors are used by the optimization algorithms to sample values or adjust
the exploration.

.. _search-space-priors:

Priors
======

We support all of `scipy distributions`_ out of the box. With the exception of the functions
outlined below, every distribution in ``scipy.stats`` can be used using the same original function
signature.

.. _search-space-prior-uniform:

``uniform(low, high)``
----------------------

The uniform distribution is redefining scipy's function signature as
``uniform(low, high)`` instead ``uniform(low, interval)``. This is to uniformize the interface with
numpy and python's builtin ``random.uniform``.

.. _search-space-prior-loguniform:

``loguniform(low, high)``
-------------------------

A logarithmic version of the uniform distribution.

.. _search-space-prior-normal:

``normal(loc, scale)``
----------------------

A wrapper for ``scipy.stats.norm``.

.. _search-space-prior-gaussian:

``gaussian(loc, scale)``
------------------------

An additional wrapper for ``scipy.stats.norm``.

.. _search-space-prior-categorical:

``choices(options)``
--------------------

``options`` may be a ``list`` of builtin python objects or a ``dict`` of builtin python objects with
their corresponding probabilities.  When passing a ``list``, each object has an equal probability of
being sampled ``choices(['all', 'equally', 'likely'])``.
To give different probabilities: ``choices({'likely': 0.8, 'unlikely': 0.2, 'impossible': 0})``.
You can pass strings, integers and floats, and all mixed together if needed:
``choices([1.0, 2, 'three'])``.

.. _search-space-prior-fidelity:

``fidelity(low, high, base=2)``
-------------------------------

This prior is a special placeholder to define a ``Fidelity`` dimension. The algorithms will not use
this prior to sample, but rather for multi-fidelity optimization. For an example of an algorithm
using multi-fidelity, you can look at the documentation of :ref:`ASHA`. To define the space of
multi-fidelity optimization, you must give the minimum (``low``) and maximum (``high``) values for
the fidelity and optionally the logarithmic base that should be used to explore this space.

.. _scipy distributions: https://docs.scipy.org/doc/scipy/reference/stats.html

Dimension Types
===============

.. py:currentmodule:: orion.algo.space

The dimensions are casted to special types according to their prior. This is critical to
either allow algorithms to leverage type information
(ex: some algorithms works better on integers) or automatically transform trial types
to make them compatible with specific algorithms
(ex: some algorithms cannot work on categorical values).

Real
----

All continuous priors are automatically casted to :class:`Real`.

.. _integer-dim:

Integer
-------

Discrete distributions of scipy are automatically casted to :class:`Integer`. All other
distributions can be casted to :class:`Integer` by setting ``discrete=True``
(ex: ``uniform(low, high, discrete=True)``).

.. warning::

   We recommend using continuous priors with ``discrete=True``
   as there is an issue with scipy discrete distribution because of incorrect interval. Issue
   is documented
   `here <https://github.com/Epistimio/orion/issues/49>`_.

Categorical
-----------

Distribition of k possible categories, with no ordinal relationship. Only the prior
``choices(options)`` is casted to :class:`Categorical`.

Fidelity
--------

Special placeholder to represent a fidelity dimension. Only the prior
:ref:`search-space-prior-fidelity` is casted to :class:`Fidelity`.

Special arguments
=================

.. _search-space-discrete:

``discrete``
------------

ex: ``uniform(0, 10, discrete=True)``

Argument to cast a continuous distribution into :ref:`integer-dim`. Defaults to ``False``.

.. _search-space-default:

``default_value``
-----------------

ex: ``uniform(0, 10, default_value=5)``

Dimensions can be set to a default value so that commandline call `insert` can support insertion
without specifying this hyperparameter, assigning it the default value. This is also useful in when
using the :ref:`EVC system`, so that experiments where an hyperparameter is deleted or added can
adapt trials from other experiments by using the default value.

.. _search-space-precision:

``precision``
-------------

ex: ``loguniform(1e-5, 1e-2, precision=2)``

Argument to sample a continuous distribution up to the requested precision. Defaults to ``4``.
The above example would sample values such as 6.7e-4 or 2.9e-3 but not 6.789e-4.


.. _search-space-shape:

``shape``
---------

ex: ``uniform(0, 10, shape=2)``

Some hyper-parameters may have multiple dimensions. This can be set using ``shape``

Configuration
=============

You can configure the search space of your experiment on the commandline call directly or
in a configuration file used by your script.

Commandline
-----------

Any argument in commandline with the form ``--arg~aprior(some, args)`` will be detected as a search
space dimension by Oríon. You can also use the verbose format ``--arg 'orion~aprior(some, args)'``.
Note that some shells may not play nicely with the parenthesis. You can format your command in the
following way to avoid this problem ``--arg~'aprior(some, args)'``.

Configuration file
-------------------

You can use configuration files to define search space with placeholder
``'orion~dist(*args, **kwargs)'`` in yaml and json files or
``name~dist(*args, **kwargs)`` in any other text-based file.

By default Oríon will only consider the file passed through the argument ``--config`` as a
configuration file for the user script. To change this behavior, take a look at the documentation
:ref:`here <customization>`.
This should not be confused with the argument ``--config`` of ``orion hunt``,
which is the configuration of Oríon. We are here referring the configuration of the user script,
represented with ``my_script_config.txt`` in the following example.

.. code-block:: console

   orion hunt --config my_orion_config.yaml ./my_script --config my_script_config.txt

During the execution of the optimization, Oríon will generate a new file for the configuration
in which the priors `orion~<prior>(<args>)` will be replaced by the actual value of the current
trial. The path of this configuration file will be passed to your script instead of the original
path. For instance, this would lead to the script from previous example to be called like this:

.. code-block:: console

   ./my_script --config /path/to/exp/working/dir/trial_random_name.conf

Here is an example of a configuration file with yaml

.. code-block:: yaml

    lr: 'orion~loguniform(1e-5, 1.0)'
    model:
      activation: "orion~choices(['sigmoid', 'relu'])"
      hiddens: 'orion~randint(100, 1000)'

Here is another example with json

.. code-block:: json

    {
      "lr": "orion~loguniform(1e-5, 1.0)"
      "model": {
        "activation": "orion~choices(['sigmoid', 'relu'])"
        "hiddens": "orion~randint(100, 1000)"
      }
    }

And here is an example with python! Note that for other files than for json and yaml, the
placeholders must be defined as ``name~dist(*args, **kwargs)``. Also, note that the code cannot be executed as is,
but once Oríon makes the substitution it will.

.. code-block:: python

    def my_config():
        lr = lr~loguniform(1e-5, 1.0)
        activations = model/activations~choices(['sigmoid', 'relu'])
        nhiddens = model/hiddens~randint(100, 1000)

        layers = []
        for layer in range(model/nlayers~randint(3, 10)):
            nhiddens /= 2
            layers.append(nhiddens)

        return lr, layers

Oríon could generate a script like this one for instance.

.. code-block:: python

    def my_config():
        lr = 0.001
        activations = 'relu'
        nhiddens = 100

        layers = []
        for layer in range(4):
            nhiddens /= 2
            layers.append(nhiddens)

        return lr, layers

When a trial is executed, a copy of the configuration file is created inside ``trial.working_dir``
and the corresponding path is passed to the user script instead of the original path.

Notes
======

Transformations
---------------

Some algorithms only support limited types of dimensions. In such case, these algorithms define
the type required, and then a wrapper transforms the space to make it compatible.

Real
~~~~

- :class:`Integer` are casted to :class:`Real`.
- :class:`Categorical` are casted to :class:`Integer` (low=0, high=number of categories)
  and then to one-hot (:class:`Real` with space=number of categories)
  to break ordinal relationship. (probabilities are lost if defined)

Integer
~~~~~~~

- :class:`Real` are quantized to :class:`Integer`.
- :class:`Categorical` are casted to :class:`Integer` (low=0, high=number of categories).
  (probabilities are lost if defined)

Conditional dependencies
------------------------

There is currently no support for conditional dependencies between dimensions.
Conditional dependencies arises in situations where some hyperparameter defines which algorithm to
use and each algorithm have its own
set of different hyperparameter. We plan to support this in the future by replacing our current
:class:`Space` implementation by `ConfigSpace`_. This should not change the current interface and
only add more special arguments. You can see the state of our plan in our `Roadmap`_.

.. _ConfigSpace: https://automl.github.io/ConfigSpace/master/
.. _Roadmap: https://github.com/Epistimio/orion/blob/master/ROADMAP.md

References
==========

- :class:`orion.core.io.space_builder.DimensionBuilder`
- :class:`orion.core.io.space_builder.SpaceBuilder`
- :class:`orion.algo.space.Space`
- :class:`orion.algo.space.Dimension`
- :class:`orion.algo.space.Real`
- :class:`orion.algo.space.Integer`
- :class:`orion.algo.space.Categorical`
- :class:`orion.algo.space.Fidelity`
