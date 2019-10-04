Results
-------

You can fetch experiments and trials using python code. There is no need to understand the
specific database backend used (such as MongoDB) since you can fetch results using the
:class:`orion.core.worker.experiment.Experiment` object.
The class :class:`orion.core.io.experiment_builder.ExperimentBuilder`
provides simple methods to fetch experiments
using their unique names. You do not need to explicitly open a connection to the database since it
will automatically infer its configuration from the global configuration file as when calling Oríon
in commandline. Otherwise you can pass other arguments to
:meth:`ExperimentBuilder().build_view_from() \
<orion.core.io.experiment_builder.ExperimentBuilder.build_view_from>`.

using the same dictionary structure as in the configuration file.

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

For a complete example, here is how you can fetch trials from a given experiment.

.. code-block:: python

   import datetime
   import pprint

   from orion.core.io.experiment_builder import ExperimentBuilder

   some_datetime = datetime.datetime.now() - datetime.timedelta(minutes=5)

   experiment = ExperimentBuilder().build_view_from({"name": "orion-tutorial"})

   pprint.pprint(experiment.stats)

   for trial in experiment.fetch_trials():
       print(trial.id)
       print(trial.status)
       print(trial.params)
       print(trial.results)
       print()
       pprint.pprint(trial.to_dict())

   # Fetches only the completed trials
   for trial in experiment.fetch_trials_by_status('completed'):
       print(trial.objective)

<<<<<<< HEAD

:class:`<orion.core.worker.experiment.Experiment>` has many methods that allows you to query
for different trials. You can find them in the code
reference section.
=======
:class:`<orion.core.worker.experiment.Experiment>` has many methods that allows you to query
for different trials. You can find them in the code
>>>>>>> 31b8e1b78329b5b0f3cd38fedc85224b16440d27

.. _`mongodb-like syntax`: https://docs.mongodb.com/manual/reference/method/db.collection.find/
