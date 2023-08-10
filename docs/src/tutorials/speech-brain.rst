********************
SpeechBrain
********************

In this short tutorial, we're going to demonstrate how Oríon can be integrated to a `SpeechBrain
<https://speechbrain.github.io/>`_ speech recognition model.
The files mentioned in this tutorial are available in the `Oríon
<https://github.com/Epistimio/orion/tree/develop/examples>`_ repository.

Installation and setup
======================

Make sure Oríon is installed (:doc:`/install/core`).

Then install SpeechBrain using ``$ pip install speechbrain``

Code used in this tutorial
==========================

In this tutorial, we are going to use some code from the `SpeechBrain
<https://github.com/speechbrain/speechbrain>` repository. More specifically, a speech recognition
template made as an example. We will repurpose this example to adapt it for Oríon. The template
used for creating this tutorial can be found `here
<https://github.com/speechbrain/speechbrain/tree/develop/templates/speech_recognition/ASR>`.
You can also directly see the code modified for this example here :
``examples/speechbrain_tutorial``.

We used the ``train.py`` file, but created a ``main.py``, with the ``main`` function,
which we slightly modified for optimizing the hyperparamers with Oríon.

Adapting the Speechbrain for Oríon
==================================

The Adaptation for using Oríon is quite simple.

1) We first need to import ``orion.report_objective()`` into the project.

.. code-block:: python

    from orion.client import report_objective

2) We then need to change the evaluation from the training data to the validation data.
The evaluation method should look like this. It returns the validation loss.

.. literalinclude:: /../../examples/speechbrain_tutorial/main.py
   :language: python
   :lines: 75-80

3) Finally, we call ``report_objective`` at the end to return the final objective value,
the validation loss, to Oríon.

.. code-block:: python

    report_objective(valid_stats)

The code is now adapted and ready to be used with Oríon.

Execution
=========

We are now going to call the ``orion hunt`` command.
Notice that we still need to give the ``train.yaml``
file to speechbrain, since the general configuration is in there. However, we are going to specify
the hyperparameters that we want to optimize in the command line,
which will automatically overrides the ones set in the ``train.yaml``. When an argument
is defined both in the yaml configuration file and in command line, SpeechBrain
gives precedence to values provided in command line. Thus, defining the hyperparamers through
the command line for Oríon allows overriding the values in ``train.yaml`` in SpeechBrain.

.. code-block:: bash

    orion hunt \
      --enable-evc -n <experiment_name> \
      python main.py train.yaml \
        --lr~'loguniform(0.05, 0.2)' \
        --ctc_weight~'loguniform(0.25, 0.75)' \
        --label_smoothing~'loguniform(1e-10, 10e-5)' \
        --coverage_penalty~'loguniform(1.0, 2.0)' \
        --temperature~'loguniform(1.0, 1.5)' \
        --temperature_lm~'loguniform(1.0, 1.5)'

Results
=======

When an experiment reaches its termination criterion, basically ``max-trials``,
you can see the results using the following command:

.. code-block:: bash

    $ orion info -n <experiment_name>

Which outputs the following statistics:

.. code-block:: bash

    Stats
    =====
    completed: True
    trials completed: 209
    best trial:
      id: 8675cfcfba768243e1ed1ac7825c69b6
      evaluation: 0.13801406680803444
      params:
        /coverage_penalty: 1.396
        /ctc_weight: 0.389
        /label_smoothing: 2.044e-10
        /lr: 0.06462
        /temperature: 1.175
        /temperature_lm: 1.087
    start time: 2022-09-29 14:37:41.048314
    finish time: 2022-09-30 20:08:07.384765
    duration: 1 day, 5:30:26.336451
