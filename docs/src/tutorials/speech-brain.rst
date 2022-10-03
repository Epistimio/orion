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

Adapting the Speechbrain for Oríon
=======================

The Adaptation for using Oríon is quite simple.

1) We first need to import orion.report_objective() into the project. 

.. code-block:: python

    from orion.client import report_objective

2) We then need to change the evaluation from the training data to the validation data. The evaluation method should look like this. It returns the validation loss.

.. literalinclude:: /../../examples/speechbrain_tutorial/main.py
   :language: python
   :lines: 75-80

3) Finally, we call ``report_objective`` at the end to return the final objective value, the validation loss, to Oríon.

.. code-block:: python
    report_objective(valid_stats)

The code is now adapted and ready to be used with Oríon.

Execution
=========

We are now going to call the orion hunt function. Notice that we still need to give the train.yaml file to speechbrain, since the general configuration is in there.
However, we are going to specify the hyper-parameters that we want to optimize after that, which will automatically overrides the ones set in the train.yaml.

orion hunt \
  --enable-evc -n <experiment_name> \ 
  python train.py train.yaml \
    --lr~'loguniform(0.05, 0.2)' \
    --ctc_weight~'loguniform(0.25, 0.75)' \
    --label_smoothing~'loguniform(1e-10, 10e-5)' \
    --coverage_penalty~'loguniform(1.0, 2.0)' \
    --temperature~'loguniform(1.0, 1.5)' \
    --temperature_lm~'loguniform(1.0, 1.5)'
