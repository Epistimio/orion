********
Optimize
********

.. contents::
   :depth: 2
   :local:


There are two ways of using Oríon for optimization. One is using the commandline interface which
conveniently turn a simple script into a hyper-parameter optimization process at the level of the
command line.
The other is using the python interface which gives total control
over the pipeline of optimization.

Commandline API
===============

Suppose you normally execute your script with the following command

.. code-block:: bash

    $ python main.py --lr 0.1

Using the commandline API you can turn your script into a hyper-parameter process by wrapping it
with Oríon.

.. code-block:: bash

    $ orion hunt -n exp-name python main.py --lr~'loguniform(1e-5, 1.0)'

An experiment called ``exp-name`` will now be created and your script will be called with
the argument ``--lr`` assigned to values sampled by the optimization algorithm.

Configuration of the algorithm can be done inside a yaml file passed to ``--config`` as described in
:ref:`Setup Algorithms`.

To return the results to orion, you must add a call to
:py:func:`orion.client.report_objective(value) <orion.client.cli.report_objective>`
in your script at the end of the execution.

See :py:mod:`orion.client.cli` for more information on all helper functions available.


Python APIs
===========

The python API is declined in two versions

:ref:`sequential_api`:
   A simple method for local sequential optimision.
:ref:`service_api`:
   A simple method to get new parameters to try and report results in a distributed manner.
:ref:`framework_api`:
   Total control over the hyper-parameter optimization pipeline.


.. _sequential_api:

Sequential API
--------------

Using the helper :py:func:`orion.client.workon`,
you can optimize a function with a single line of code.

.. code-block:: python

   from orion.client import workon


   def foo(x):
      return [dict(
         name='dummy_objective',
         type='objective',
         value=1)]


   experiment = workon(foo, space=dict(x='uniform(-50,50)'))


The experiment object returned is can be used to fetch the database of trials
and analyze the optimization process. Note that the storage for `workon` is
in-memory and requires no setup. This means however that :py:func:`orion.client.workon`
cannot be used for parallel optimisation.

.. _service_api:

Service API
-----------

Experiments are created using the helper function
:py:func:`orion.client.create_experiment`.
You can then sample new trials with
:py:meth:`experiment.suggest() <orion.client.experiment.ExperimentClient.suggest>`.
The parameters of the trials are provided as a dictionary with
:py:meth:`trial.params <orion.core.worker.trial.Trial.params>`.
Once the trial is completed, results can be reported to the experiment with
:py:meth:`experiment.observe() <orion.client.experiment.ExperimentClient.observe>`.
Note that this should be the final result of the trial. When observe is called, the trial
reservation is released and its status is set to completed. Observing twice the same trial will
raise a RuntimeError because the trial is not reserved anymore.

.. code-block:: python

   from orion.client import create_experiment

   experiment = create_experiment(
      name='foo',
      space=dict(x='uniform(-50,50)'))

   trial = experiment.suggest()

   # Do something using trial.params['x']

   results = [dict(
       name='dummy_objective',
       type='objective',
       value=dummy_objective)]

   experiment.observe(trial, results)


The storage used by the experiment can be specified as an argument to
:py:func:`create_experiment(storage={}) <orion.client.create_experiment>`
or in a global configuration file as described in :ref:`install_database`.

To distribute the hyper-parameter optimisation in many workers, simply execute your script in
parallel where you want to execute your trials. The method
:py:meth:`experiment.suggest() <orion.client.experiment.ExperimentClient.suggest>`
will take care of synchronizing the local algorithm with all remote instances, making it possible
to distribute the optimization without setting up a master process.

See :py:class:`ExperimentClient <orion.client.experiment.ExperimentClient>`
for more information on the experiment client object.

.. warning::

   Code version detection is not currently supported. This means that creating experiments using
   different code version will not lead to version increment like it would do with the commandline
   API.


.. _framework_api:


Framework API
-------------

.. warning::

   This API is not implemented yet. It should be included in v0.2.0.

.. code-block:: python

   from orion.client import create_space
   from orion.client import create_algo

   space = create_space(x='uniform(-50,50)')

   algo = create_algo(space, type='ASHA', add some config here)

   params = algo.suggest()

   results = 'some_results...'

   algo.observe(params, results)
