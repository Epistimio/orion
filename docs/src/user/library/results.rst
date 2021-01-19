Results
-------

You can fetch experiments and trials using python code. There is no need to understand the
specific database backend used (such as MongoDB) since you can fetch results using the
:class:`orion.client.experiment.ExperimentClient` object.
The helper function :py:func:`orion.client.get_experiment`
provides a simple way to fetch experiments
using their unique names. You do not need to explicitly open a connection to the database since it
will automatically infer its configuration from the global configuration file as when calling Oríon
in commandline. Otherwise you can specify the configuration directly to
:py:func:`get_experiment() <orion.client.get_experiment>`. Take a look at the documentation
for more details on all configuration arguments that are supported.

.. code-block:: python

   # Database automatically inferred
   get_experiment(name="orion-tutorial")

   # Database manually set
   get_experiment(
       name="orion-tutorial",
       storage={
           'type': 'legacy',
           'database': {
               'type': 'mongodb',
               'name': 'myother',
               'host': 'localhost'}})

For a complete example, here is how you can fetch trials from a given experiment.

.. code-block:: python

   import pprint

   from orion.client import get_experiment

   experiment = get_experiment(name="orion-tutorial")

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


:py:class:`ExperimentClient <orion.client.experiment.ExperimentClient>`
has many methods that allows you to query
for different trials. You can find them in the code reference section.
