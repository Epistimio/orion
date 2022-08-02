``info`` Detailed information about experiments
------------------------------------------------

This command gives a detailed description of a given experiment.
Here is an example of all the sections provided by the command.

.. code-block:: console

   orion info --name orion-tutorial

.. code-block:: bash

   Identification
   ==============
   name: orion-tutorial
   version: 1
   user: <username>

   Commandline
   ===========
   python main.py --lr~loguniform(1e-5, 1.0)


   Config
   ======
   pool size: 1
   max trials: inf


   Algorithm
   =========
   random:
       seed: None


   Space
   =====
   /lr: reciprocal(1e-05, 1.0)


   Meta-data
   =========
   user: <username>
   datetime: 2019-07-18 18:57:57.840000
   orion version: v0.1.5


   Parent experiment
   =================
   root:
   parent:
   adapter:


   Stats
   =====
   trials completed: 1
   best trial:
     /lr: 0.03543491957849911
   best evaluation: 0.9626
   start time: 2019-07-18 18:57:57.840000
   finish time: 2019-07-18 18:58:08.565000
   duration: 0:00:10.725000


The last section contains information about the best trial so far, providing its
hyperparameter values and the corresponding objective.

The ``--name`` argument
~~~~~~~~~~~~~~~~~~~~~~~
To get information on an experiment, you need to call `info` with the `--name` or `-n` argument like
shown in the previous example. This will fetch the latest version of the experiment with that name
inside the database and display its content.

The ``--version`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~
To specify which version of an experiment you wish to observe, you can use the `--version` argument.
If provided, this will fetch the experiment with a version number corresponding to that version
instead of fetching the latest one. Note that the `--version` argument cannot be used alone and that
an invalid version number, i.e. a version number greater than the latest version, will simply fetch
the latest one instead.

For example, suppose we have two experiments named `orion-tutorial` inside the database, one with
version `1` and one with version `2`. Then running the following command would simply give us the
latest version, so version `2`.

.. code-block:: console

   orion info --name orion-tutorial

Whereas, running the next command will give us the first version instead:

.. code-block:: console

   orion info --name orion-tutorial --version 1
