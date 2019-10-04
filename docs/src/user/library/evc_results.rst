Iterative Results with EVC
--------------------------

When using the experiment version control (described `here <user/evc>`_),
the experiments are connected in a tree structure which we call the EVC tree.
You can retrieve results from different experiments with the EVC tree similarly
as described in previous section. The only difference
is we need to use :class:`EVCBuilder <orion.core.io.evc_builder.EVCBuilder>` instead of
:class:`ExperimentBuilder <orion.core.io.experiment_builder.ExperimentBuilder>`.
The :class:`EVCBuilder <orion.core.io.evc_builder.EVCBuilder>` will connect the experiment
to the EVC tree, accessible through the
:attr:`node <orion.core.worker.experiment.Experiment.node>` attribute.
All trials of the tree can be fetched
with option
:meth:`fetch_trials(with_evc_tree=True) <orion.core.worker.experiment.Experiment.fetch_trials>`,
``with_evc_tree=False``` will only fetch the
trials of the specific experiment.

.. code-block:: python

   import pprint
   from orion.core.io.evc_builder import EVCBuilder

   experiment = EVCBuilder().build_view_from(
       {"name": "orion-tutorial-with-momentum"})

   print(experiment.name)
   pprint.pprint(experiment.stats)

   parent_experiment = experiment.node.parent.item
   print(parent_experiment.name)
   pprint.pprint(parent_experiment.stats)

   for child in experiment.node.children:
       child_experiment = child.item
       print(child_experiment.name)
       pprint.pprint(child_experiment.stats)
