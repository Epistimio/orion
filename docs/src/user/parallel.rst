.. _parallelism:

****************
Parallel Workers
****************

In this chapter, we describe how Oríon can be run on multiple cores or computers for the same
optimization experiments.

In most frameworks, a master-workers architecture is used. This implies that the master process must
be instantiated either by the user or by a third party provider, which incurs a significant
overhead for the users and third party dependencies -- often requiring to have an internet
connection.

Oríon has a different approach that nullify these issues: we don't have a master process. Instead,
the workers make decisions based on their shared common history stored in the database. The
operations in the database are non-blocking, ensuring horizontal scalability for large search
spaces.

We illustrate below the workflow for an hyperparameter optimization with a single worker, typically
executed on a personal laptop.

.. figure:: /_resources/one.png
  :alt: A single worker optimizing an experiment.
  :align: center
  :figclass: align-center

More workers can be invoked by simply running the ``$ orion hunt -n exp ...`` command multiple
times. Each call spawns a new worker for the given experiment. The workers' workflow is unchanged
because the workers are synchronized during the creation of a new trial based on what other trials
were already completed by other workers.

.. figure:: /_resources/synchronization.png
  :alt: Multiple workers are synchronized while creating a new trial.
  :align: center
  :figclass: align-center


Executor backends
=================

It is also possible to execute multiple workers using the argument ``--n-workers`` in commandline
or ``experiment.workon(n_workers)`` using the python API. The workers will work together
using the same mechanisms explained above, but an
:class:`orion.executor.base.BaseExecutor` backend will be used in addition
to spawn the workers and maintain them alive. The default backend is :ref:`executor-joblib`.

You can configure it
via a global or a local configuration file (:ref:`config_worker_executor_configuration`)
or by passing it as an argument to :py:meth:`orion.client.experiment.ExperimentClient.tmp_executor`.

.. _executor-joblib:

Joblib
------

`Joblib`_ is a lightweight library for task parallel execution in Python. We use the ``loky``
backend of joblib by default to spawn workers on different processes.
The joblib backend is configured using `parallel_backend()`_.

See documentation at `parallel_backend()`_ for information on possible arguments.

.. _Joblib: https://joblib.readthedocs.io/en/latest/

.. _parallel_backend(): https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend

Dask
----

It is possible to use dask with the joblib backend. Joblib can be configured to use Dask as
explained
`here <https://joblib.readthedocs.io/en/latest/auto_examples/parallel/distributed_backend_simple.html>`__.
For more control over Dask, you should prefer using Dask executor backend directly.
The executor configuration is used to create the Dask Client. See Dask's documentation
`here <https://distributed.dask.org/en/latest/api.html#distributed.Client>`__ for
more information on possible arguments.
