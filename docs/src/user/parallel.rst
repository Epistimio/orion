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
