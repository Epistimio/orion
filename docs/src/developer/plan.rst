***************
Source code map
***************

This document will walk the path of an orion experiment through the
code. Not every detail is explained, but there are ample links to the
classes and methods involved if you want to dig further in a certain
section.

Departure
---------

You start an experience by running ``orion hunt <script> <params>``.

The code in :py:func:`orion.core.cli.main` will parse the command line
arguments and route to :py:func:`orion.core.cli.hunt.main`.

The command line arguments are passed to
:py:func:`orion.core.io.experiment_builder.build_from_args`, which
does some setup and hands over the arguments to
:py:func:`orion.core.io.experiment_builder.build`. This will hand over
the configuration to
:py:func:`orion.core.io.experiment_builder.consolidate_config` which
will look up the experiment in the configured storage to see if it's
already there and merge the loaded configuration with the provided one
with various helpers from :py:mod:`orion.core.io.resolve_config` to
build the final configuration. The result is eventually handled off to
:py:func:`orion.core.io.experiment_builder.create_experiment` to
create an :py:class:`orion.core.worker.experiment.Experiment` and set
its properties.

If the experiment is new, meaning it has no storage id, then it will
attempt to save it to storage, which may conflict in case another
instance of ``orion hunt`` is doing the same thing. The storage is
responsible for reporting conflicts and
:py:func:`orion.core.io.experiment_builder.build` is called again
recursively in that case to retry the whole operation.

The created experiment finds its way back to
:py:func:`orion.core.cli.hunt.main` and is handed off to
:py:func:`orion.core.cli.hunt.workon` along with some more
configuration for the workers.

This method will setup a few more objects to manage the optimization
process: a :py:class:`orion.core.worker.consumer.Consumer` to act as
the bridge to the user script and an
:py:class:`orion.client.experiment.ExperimentClient` to coordinate
everything and calls
:py:meth:`orion.client.experiment.ExperimentClient.workon` which
mostly creates a :py:class:`orion.client.runner.Runner` and calls its
:py:meth:`orion.client.runner.Runner.run` method.


The Run Loop
------------

We are finally in the main run loop. It is composed of three main
phases that repeat.


First phase
~~~~~~~~~~~

In the first phase we call
:py:meth:`orion.client.runner.Runner.sample`. This will check if new
trials are required using
:py:meth:`orion.client.runner.Runner.should_sample` and request those
trials using :py:meth:`orion.client.experiment.ExperimentClient.suggest`.

This will first check if any trials are available in the storage using
:py:meth:`orion.core.worker.experiment.Experiment.reserve_trial`.

If none are available, it will produce new trials using
:py:meth:`orion.core.worker.producer.Producer.produce` which loads
the state of the algorithm from the storage, runs it to suggest new
:py:class:`orion.core.worker.trial.Trial` and saves both the new
trials and the new algorithm state to the storage. This is protected
from concurrent access by other instances of ``orion hunt`` by locking
the storage for the duration of that operation.


The second phase
~~~~~~~~~~~~~~~~

In the second phase we call
:py:meth:`orion.client.runner.Runner.scatter` with the trials
generated in the first phase, if any.

This schedules each trial to be executed using the configured executor
and registers the futures that the executor returns. Execution is
handled asynchronously and the futures enable us to keep track of the
state of the trials.


The third phase
~~~~~~~~~~~~~~~

In the third phase we call
:py:meth:`orion.client.runner.Runner.gather` which will wait on all
currently registered futures with a timeout to get some results.

Once we get those results we de-register the futures and record the
results with
:py:meth:`orion.client.experiment.ExperimentClient.observe` or update
the count of broken trials if they did not finish successfully.

Finally we monitor the total amount of time spent waiting for trials
to finish.


Stopping criteria
~~~~~~~~~~~~~~~~~

There are multiple criteria that are monitored to stop the
experiment.

The first obvious one is the configured maximum number of trials to
run. If this is reached, then we stop running more. This is checked at
the beginning of the loop with
:py:attr:`orion.client.runner.Runner.is_running`.

The experiment can also stop if too many trials fail, either because
they fail to start, they crashed, were killed (like by an external job
scheduler) or the take too much time to complete. This is checked in
:py:meth:`orion.client.runner.Runner.gather` with
:py:attr:`orion.client.runner.Runner.is_broken`.

If one of the workers returns an unexpected result the experiment is
also stop immediately because it is assume that something is wrong
with either the code or the configuration and spending more time
computing stuff will not fix it. This is also checked for in
:py:meth:`orion.client.runner.Runner.gather`.

Finally if the loop spends too much time waiting and nothing happens
the experiment is considered stalled and will also stop. This is
checked at the end of :py:meth:`orion.client.runner.Runner.run`.
