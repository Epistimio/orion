*****************************************************************************************
Hyperparameters optimisation using a HuggingFace Model and the Hydra-Orion-Sweeper plugin
*****************************************************************************************

In this tutorial, we will show an easy Orion integration of a HuggingFace model, using Hydra, with the 
`Hydra_Orion_Sweeper <https://github.com/Epistimio/hydra_orion_sweeper >`_ plugin. Hydra is essentially a framework for
configuring applications. We will use it to define our Hyperparameters and some Orion configuration.

Imports
^^^^^^^
You will need to import some modules using pip, such as ``datasets``, ``transformers``, ``orion`` and ``hydra-orion-sweeper``.

 .. literalinclude:: /../../examples/huggingface/main.py
   :language: python
   :lines: 8-25


Hydra configuration file
^^^^^^^^^^^^^^^^^^^^^^^^

Notice here how the arguments that are not defined will be set at None, which will be overridden by default values
or not used at all. This serves as a replacement for parsing arguments in the command-line, but is integrated with Orion,
which makes the usability really good.

 .. literalinclude:: /../../examples/huggingface/config.yaml
   :language: yaml
   :lines: 1-

You can find more about the Hydra-Orion-Sweeper plugin by looking directly at the 
Github Repository : `Hydra_Orion_Sweeper <https://github.com/Epistimio/hydra_orion_sweeper >`_ , or find out more
about Hydra in general here : `Hydra_Orion_Sweeper <https://github.com/Epistimio/hydra_orion_sweeper >`_


Adapting the code for Hydra
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``config_path`` and ``config_name`` here specifies the path of your Hydra file.
The ``cfg.args`` are reference to the ``args`` in the config file.
We also are going to be using this main function as the entry point of the program. 

 .. literalinclude:: /../../examples/huggingface/main.py
   :language: python
   :lines: 136-139, 240-241


Adapting the code for Oríon
^^^^^^^^^^^^^^^^^^^^^^^^^^^
for Orion, we only need to make one small change to the script: we report to Oríon the objective that we
want to **minimize** at the end of the script using :py:func:`orion.client.report_objective`:

.. literalinclude:: /../../examples/huggingface/main.py
   :language: python
   :lines: 227-230


:py:func:`orion.client.report_objective` can be imported using :

.. code-block:: python

   from orion.client import report_objective

Execution
^^^^^^^^^
We simply have to run the main.py file with the -m argument, which makes sure we use the Hydra-Orion-Sweeper plugin.

.. code-block:: bash

   $ python3 main.py -m