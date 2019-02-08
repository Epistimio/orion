**************
Simple example
**************

Installation and setup
======================

In this tutorial you will run a very simple MNIST example in pytorch using Oríon.
First, install Oríon follwing :doc:`/install/core` and configure the database
(:doc:`/install/database`). Then install `pytorch`, `torchvision` and clone the
PyTorch [examples repository](https://github.com/pytorch/examples):

.. code-block:: bash

    $ pip3 install torch torchvision
    $ git clone git@github.com:pytorch/examples.git


.. _examples repository: https://github.com/pytorch/examples


Adapting the code of MNIST example
==================================
After cloning pytorch examples repository, cd to mnist folder:

.. code-block:: bash

    $ cd examples/mnist

In your favourite editor add a shebang line `#!/usr/bin/env python` to
the `main.py` and make it executable, for example:

.. code-block:: bash

    $ sed -i '1s/^/#!/usr/bin/env python/' main.py
    $ chmod +x main.py

At the top of the file, below the imports, add one line of import the helper function
orion.client.report_results():

.. code-block:: python

    from orion.client import report_results

We are almost done now. We need to add a line to the function `test()` so that
it returns the error rate.

.. code-block:: python

    return 1 - (correct / len(test_loader.dataset))

And finally, we get back this test error rate and call ``report_results`` to
return the objective value to Oríon. Note that ``report_results`` is meant to
be called only once, this is because Oríon only optimizes looking at 1
objective value.

.. code-block:: python

        test_error_rate = test(args, model, device, test_loader)

    report_results([dict(
        name='test_error_rate',
        type='objective',
        value=test_error_rate)])

You can also return result types of ``'gradient'`` and ``'constraint'`` for
algorithms which supports those results as well.

Important note here, we use test error rate for sake of simplicity, because the
script does not contain validation dataset loader as-is, but we should
**never** optimize our hyper-parameters on the test set. We should always use a
validation set.

Another important note, Oríon will always **minimize** the objective so make sure you never try to
optimize something like the accuracy of the model unless you are looking for very very bad models.


Execution
=========

Once the script is adapted, optimizing the hyper-parameters with Oríon is
rather simple. Normally you would call the script the following way.

.. code-block:: bash

    $ ./main.py --lr 0.01

To use it with Oríon, you simply need to prepend the call with
``orion hunt -n <some name>`` and specify the hyper-parameter prior
distributions.

.. code-block:: bash

    $ orion hunt -n orion-tutorial ./main.py --lr~'loguniform(1e-5, 1.0)'

This commandline call will sequentially execute ``./main.py --lr=<value>`` with random
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

    $ orion init_only -n orion-tutorial ./main.py --lr~'loguniform(1e-5, 1.0)'


Debugging
=========

When preparing a script for hyper-parameter optimization, we recommend first testing with ``debug``
mode. This will use an in-memory database which will be flushed at the end of execution. If you
don't use ``--debug`` you will likely quickly fill your database with broken experiments.

.. code-block:: bash

    $ orion --debug hunt -n orion-tutorial ./main.py --lr~'loguniform(1e-5, 1.0)'

Hunting Options
---------------

.. code-block:: bash

    $ orion hunt --help

    Oríon arguments (optional):
      These arguments determine orion's behaviour

      -n stringID, --name stringID
                            experiment's unique name; (default: None - specified
                            either here or in a config)
      -c path-to-config, --config path-to-config
                            user provided orion configuration file
      --max-trials #        number of jobs/trials to be completed (default:
                            inf/until preempted)
      --pool-size #         number of concurrent workers to evaluate candidate
                            samples (default: 10)

``name``

The unique name of the experiment.

``config``

Configuration file for Oríon which may define the database, the algorithm and all options of the
command hunt, including ``name``, ``pool-size`` and ``max-trials``.

``max-trials``

The maximum number of trials tried during an experiment.

``pool-size``

The number of trials which are generated by the algorithm each time it is interrogated.


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


You can also fetch the results using python code. You do not need to understand MongoDB since
you can fetch results using the ``Experiment`` object. The class `ExperimentBuilder` provides
simple methods to fetch experiments using their unique names. You do not need to explicitly
open a connection to the database when using the `ExperimentBuilder` since it will automatically
infer its configuration from the global configuration file as when calling Oríon in commandline.
Otherwise you can pass other arguments to ``ExperimentBuilder().build_view_from()`` using the same
dictionary structure as in the configuration file.

.. code-block:: python

   # Database automatically inferred
   ExperimentBuilder().build_view_from(
       {"name": "orion-tutorial"})

   # Database manually set
   ExperimentBuilder().build_view_from(
       {"name": "orion-tutorial",
        "dataset": {
            "type": "mongodb",
            "name": "myother",
            "host": "localhost"}})

For a complete example, here's how you can fetch trials from a given experiment.

.. code-block:: python

   import datetime
   import pprint

   from orion.core.io.experiment_builder import ExperimentBuilder

   some_datetime = datetime.datetime.now() - datetime.timedelta(minutes=5)

   experiment = ExperimentBuilder().build_view_from({"name": "orion-tutorial"})

   pprint.pprint(experiment.stats)

   for trial in experiment.fetch_trials({}):
       print(trial.id)
       print(trial.status)
       print(trial.params)
       print(trial.results)
       print()
       pprint.pprint(trial.to_dict())

   # Fetches only the completed trials
   for trial in experiment.fetch_trials({'status': 'completed'}):
       print(trial.objective)

   # Fetches only the most recent trials using mongodb-like syntax
   for trial in experiment.fetch_trials({'end_time': {'$gte': some_datetime}}):
       print(trial.id)
       print(trial.end_time)

You can pass queries to ``fetch_trials()``, where queries can be a simple dictionary of values to
match like ``{'status': 'completed'}``, in which case it would return all trials where
``trial.status == 'completed'``, or they can be more complex using `mongodb-like syntax`_.

.. _`mongodb-like syntax`: https://docs.mongodb.com/manual/reference/method/db.collection.find/


