Iterative Results with EVC
--------------------------

When using the experiment version control (described :doc:`here </user/evc>`),
the experiments are connected in a tree structure which we call the EVC tree.
You can retrieve results from different experiments with the EVC tree similarly
as described in previous section. All trials of the tree can be fetched
with option
:meth:`fetch_trials(with_evc_tree=True) <orion.client.experiment.ExperimentClient.fetch_trials>`,
``with_evc_tree=False``` will only fetch the
trials of the specific experiment.

.. code-block:: python

   import pprint

   from orion.client import get_experiment

   experiment = get_experiment(name="orion-tutorial-with-momentum")

   print(experiment.name)
   pprint.pprint(experiment.stats)

   parent_experiment = experiment.node.parent.item
   print(parent_experiment.name)
   pprint.pprint(parent_experiment.stats)

   for child in experiment.node.children:
       child_experiment = child.item
       print(child_experiment.name)
       pprint.pprint(child_experiment.stats)
