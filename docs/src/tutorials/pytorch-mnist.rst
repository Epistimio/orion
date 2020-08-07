*************
PyTorch MNIST
*************

This is a simple tutorial on running hyperparameter search with Oríon on Pytorch's MNIST example

Installation and setup
======================

Make sure Oríon is installed (:doc:`/install/core`) and the database is configured
(:doc:`/install/database`).

Then install ``pytorch`` and ``torchvision`` and clone the
PyTorch `examples repository`_:

.. code-block:: bash

    $ pip3 install torch torchvision
    $ git clone https://github.com/pytorch/examples.git


.. _examples repository: https://github.com/pytorch/examples


Adapting the code for Oríon
===========================

To use Oríon with any code we need to do two things

1. import the ``orion.client.report_objective`` helper function
2. call `report_objective` on the final objective output to be minimized
   (e.g. final test error rate)

After cloning pytorch examples repository, cd to mnist folder:

.. code-block:: bash

    $ cd examples/mnist

1. At the top of the file, below the imports, add one line of import for the helper function
``orion.client.report_objective()``:

.. code-block:: python

    from orion.client import report_objective

2. We need the test error rate so we're going to add a line to the function ``test()`` to return it

.. code-block:: python

    return 1 - (correct / len(test_loader.dataset))

Finally, we get back this test error rate and call ``report_objective`` to
return the final objective value to Oríon. Note that ``report_objective`` is meant to
be called only once because Oríon only looks at 1 ``'objective'`` value per run.

.. code-block:: python

        test_error_rate = test(args, model, device, test_loader)

    report_objective(test_error_rate)


Execution
=========

Once the script is adapted, optimizing the hyper-parameters with Oríon is
rather simple. Normally you would call the script the following way.

.. code-block:: bash

    $ python main.py --lr 0.01

To use it with Oríon, you simply need to prepend the call with
``orion hunt -n <experiment name>`` and specify the hyper-parameter prior
distributions.

.. code-block:: bash

    $ orion hunt -n orion-tutorial python main.py --lr~'loguniform(1e-5, 1.0)'

This commandline call will sequentially execute ``python main.py --lr=<value>`` with random
values sampled from the distribution ``loguniform(1e-5, 1.0)``. We support all
distributions from scipy.stats_, plus ``choices()`` for categorical
hyper-parameters (similar to numpy's `choice function`_).

.. _scipy.stats: https://docs.scipy.org/doc/scipy/reference/stats.html
.. _`choice function`: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html

Experiments are interruptible, meaning that you can stop them either with
``<ctrl-c>`` or with kill signals. If your script is not resumable automatically then resuming an
experiment will restart your script from scratch.

You can resume experiments using the same commandline or simply by specifying
the name of the experiment.

.. code-block:: bash

    $ orion hunt -n orion-tutorial

Note that experiment names are unique, you cannot create two different
experiment with the same name.

You can also register experiments without executing them.

.. code-block:: bash

    $ orion hunt --init-only -n orion-tutorial python main.py --lr~'loguniform(1e-5, 1.0)'


Results
=======

When an experiment reaches its termination criterion, basically ``max-trials``, it will print the
following statistics if Oríon is called with ``-v`` or ``-vv``.

.. code-block:: bash

    RESULTS
    =======
    {'best_evaluation': 0.05289999999999995,
     'best_trials_id': 'b7a741e70b75f074208942c1c2c7cd36',
     'duration': datetime.timedelta(0, 49, 751548),
     'finish_time': datetime.datetime(2018, 8, 30, 1, 8, 2, 562000),
     'start_time': datetime.datetime(2018, 8, 30, 1, 7, 12, 810452),
     'trials_completed': 5}

    BEST PARAMETERS
    ===============
    [{'name': '/lr', 'type': 'real', 'value': 0.012027705702344259}]

These results can be printed in terminal later on with the command :ref:`info <cli-info>` or
fetched using the :ref:`library API <library-api-results>`.

.. code-block:: bash

    $ orion info -n orion-tutorial

Notes
=====
We use test error rate for sake of simplicity, because the
script does not contain validation dataset loader as-is, but we should
**never** optimize our hyper-parameters on the test set and instead always use a
validation set.

Oríon will always **minimize** the objective so make sure you never try to
optimize something like the accuracy of the model unless you are looking for very very bad models.

You can also report results of types ``'gradient'`` and ``'constraint'`` for
algorithms which require those parameters as well, or ``'statistic'`` for metrics
to be saved with the trial. See
:py:func:`report_results() <orion.client.cli.report_results>`
for more details.


Debugging
=========

When preparing a script for hyper-parameter optimization, we recommend first testing with ``debug``
mode. This will use an in-memory database which will be flushed at the end of execution. If you
don't use ``--debug`` you will likely quickly fill your database with broken experiments.

.. code-block:: bash

    $ orion --debug hunt -n orion-tutorial python main.py --lr~'loguniform(1e-5, 1.0)'

Hunting Options
---------------

.. code-block:: console

    $ orion hunt --help

    Oríon arguments (optional):
      These arguments determine orion's behaviour

      -n stringID, --name stringID
                            experiment's unique name; (default: None - specified
                            either here or in a config)
      -u USER, --user USER  user associated to experiment's unique name; (default:
                            $USER - can be overriden either here or in a config)
      -c path-to-config, --config path-to-config
                            user provided orion configuration file
      --max-trials #        number of trials to be completed for the experiment.
                            This value will be saved within the experiment
                            configuration and reused across all workers to
                            determine experiment's completion. (default: inf/until
                            preempted)
      --worker-trials #     number of trials to be completed for this worker. If
                            the experiment is completed, the worker will die even
                            if it did not reach its maximum number of trials
                            (default: inf/until preempted)
      --working-dir WORKING_DIR
                            Set working directory for running experiment.
      --pool-size #         number of simultaneous trials the algorithm should
                            suggest. This is useful if many workers are executed
                            in parallel and the algorithm has a strategy to sample
                            non-independant trials simultaneously. Otherwise, it
                            is better to leave `pool_size` to 1 and set a Strategy
                            for Oríon's producer. Note that this option is not usefull useless you
                            know the algorithm have a strategy to produce multiple trials
                            simultaneously. If you have any doubt, leave it to 1.
                            (default: 1)

``name``

The unique name of the experiment.

``user``

Username used to identify the experiments of a user. The default value is the system's username
$USER.

``config``

Configuration file for Oríon which may define the database, the algorithm and all options of the
command hunt, including ``name``, ``pool-size`` and ``max-trials``.

``max-trials``

The maximum number of trials tried during an experiment.

``worker-trials``

The maximum number of trials to be executed by a worker (a single call to ``orion hunt [...]``).

``working-dir``

The directory where configuration files are created. If not specified, Oríon will create a
temporary directory that will be removed at end of execution of the trial.

``pool-size``

The number of trials which are generated by the algorithm each time it is interrogated. This is
useful if many workers are executed in parallel and the algorithm has a strategy to sample
non-independant trials simultaneously. Otherwise, it is better to leave ``pool_size`` to its default
value 1. Note that this option is not usefull useless you know the algorithm have a strategy
to produce multiple trials simultaneously. If you have any doubt, leave it to 1. :)
