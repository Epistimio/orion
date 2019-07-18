``info`` Detailed informations about experiments
------------------------------------------------

This commands gives a detailed description of a given experiment.
Here is an example of all the sections provided by the command.

.. code-block:: console

   orion info orion-tutorial

.. code-block:: bash

   Commandline
   ===========
   --lr~loguniform(1e-5, 1.0)


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
hyper-parameter values and the corresponding objective
