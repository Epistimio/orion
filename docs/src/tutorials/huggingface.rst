*****************************************************************************************
Hyperparameters optimisation using a HuggingFace Model and the Hydra-Orion-Sweeper plugin
*****************************************************************************************

In this tutorial, we will show an easy Orion integration of a HuggingFace translation model using
Hydra, with the `Hydra_Orion_Sweeper <https://github.com/Epistimio/hydra_orion_sweeper>`_ plugin.
Hydra is essentially a framework for configuring applications. We will use it to define our
Hyperparameters and some Orion configuration. We will also be using
`Comet <https://www.comet.com/>`_ for monitoring our experiments.
Installation
^^^^^^^^^^^^
For this tutorial everything that we need to install can be can be found in the ``requirements.txt``
file located in the ``examples/huggingface`` repository. You can then install the requirements
with ``pip``.

.. code-block:: bash

   $ pip install -r examples/huggingface/requirements.txt

Imports
^^^^^^^
You will now need to import these modules.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 5-6,13-22


Hydra configuration file
^^^^^^^^^^^^^^^^^^^^^^^^

Notice here how the arguments that are not defined will be set at ``None``, which will be
overridden by default values or not used at all. This serves as a replacement for parsing arguments
in the command line, but is integrated with Orion, which makes it more practical
to manage search spaces of hyperparameters.

.. literalinclude:: /../../examples/huggingface/config.yaml
  :language: yaml
  :lines: 1-

If you want to change your working or/and logging directory, you can also easily do that.
From the config file, you can specify

.. literalinclude:: /../../examples/huggingface/config.yaml
  :language: yaml
  :lines: 29-31

This will change your working directory. You can see that with the hydra-orion-sweeper, we are able
to specify 4 variables from Orion. ``${hydra.sweeper.orion.name}`` for the ``experiment_name``,
``${hydra.sweeper.orion.id}`` for the ``experiment_id``, ``${hydra.sweeper.orion.uuid}`` for the
experiment ``uuid`` and ``${hydra.sweeper.orion.trial}`` for the ``trial_id``.

In the code, you can now specify the output directory to the trainer with the
``output_dir`` parameter. ``os.getcwd()`` specifies the current working dir.

Including these options will create different folders for each trial, under different ones for
each experiment and even different ones for each sweep. You do not have to add them all, but it
can be quite useful when you don't want 2 trials writing their cache in the same file, which
could result in an error.

.. code-block:: python

    output_dir=str(os.getcwd())+"/test_trainer",

You can find more about the Hydra-Orion-Sweeper plugin by looking directly at the
Github Repository : `Hydra_Orion_Sweeper <https://github.com/Epistimio/hydra_orion_sweeper>`_ ,
or find out more about Hydra in general here : `Hydra <https://hydra.cc/docs/intro/>`_


Adapting the code for Hydra
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``config_path`` and ``config_name`` here specifies the path of your Hydra file.
The ``cfg.args`` are reference to the ``args`` in the config file.
We also are going to be using this main function as the entry point of the program.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 141-146, 310-311

With the ``hydra-orion-sweeper``, this function needs to return the objective. You have 3 choices
for what you can return :

.. code-block:: python

    if cfg.return_type == "float":
        return out

    if cfg.return_type == "dict":
        return dict(name="objective", type="objective", value=out)

    if cfg.return_type == "list":
        return [dict(name="objective", type="objective", value=out)]

For the purpose of this tutorial, we are going to keep it simple and simply return a float,
our objective that we want to minimize.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 304

Comet
^^^^^
We are going to use Comet to track our experiments. It is quite simple to use. First,
install comet using

.. code-block:: bash

   $ pip install comet-ml

Now that it is installed, we simply have to set some environment variables, such as

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 8-13

You can also set them in your working environment. If you are to set them in python, however,
you need to make sure to set them before importing ``transformers``.

For the ``COMET_API_KEY``, you will be given a token when creating your comet account.
This is the one you are going to use here.

And that is it ! If the variables are set and comet-ml is downloaded, HuggingFace will
automatically upload your data to Comet, you simply have to go to your profile on their site
and see your experiments.

It is important to note here that we can swap the Comet logger to many others, such as WandB,
MLFlow, Neptune and ClearML. You can see the complete list in the HuggingFace documentation
`HF callbacks <https://huggingface.co/docs/transformers/main_classes/callback#callbacks>`_

Example code
^^^^^^^^^^^^
For this example, we are fine-tuning a pretrained translation model named ``Helsinki-NLP``.
We start by setting the training arguments.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 165-174

For our purposes, we will be using a ``Seq2SeqTrainer``, so for the training arguments are going
to be ``Seq2SeqTrainingArguments``. The ``set_training_args`` function adds the hydra arguments
into the training arguments.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 114-118

For the dataset, we are going to use the ``wmt16`` dataset. We can set a ``cache_dir`` to where
the dataset cache will be stored

.. literalinclude:: /../../examples/huggingface/main.py
   :language: python
   :lines: 179,182-184

We then prepare our training and evaluation datasets. In this example, we want to evaluate our
model with the validation dataset and the training dataset.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 196-229

For the metric, we are going to use ``sacrebleu``. We can also set a ``cache_dir`` here for the
metric cache files. The ``compute_metrics`` function goes as follows :

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 238-240, 247-268

Now we have to create the actual Trainer, a ``Seq2SeqTrainer`` as mentioned previously.
It is very much like a classic ``Trainer`` from HuggingFace.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 284-292

HuggingFace will log the evaluation from the ``eval_dataset`` to Comet. Since we also want the
evaluation from the training dataset, we will have to implement something called a
``CustomCallback``. The one I made for this tutorial takes the ``trainer`` and the dataset we want
to add (in our case, our train dataset) as parameters.
We can then rewrite some callback functions, such as ``on_epoch_end()``.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 270-282,294

All that is left to do now is to train the model, and once it's finish training, send the data to
Orion by returning it.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 295-297, 304

For more details, feel free to simply go look at the code, in ``examples/huggingface/main.py``

Execution
^^^^^^^^^
We simply have to run the main.py file with the -m argument, which makes sure we use the
Hydra-Orion-Sweeper plugin.

.. code-block:: bash

   $ python3 main.py -m

Visualizing results
^^^^^^^^^^^^^^^^^^^
With Orion, after your experiment has finished running, you can easily visualize your results
using `regret plots <https://orion.readthedocs.io/en/stable/auto_examples/plot_1_regret.html>`_
and `partial dependencies plots
<https://orion.readthedocs.io/en/stable/auto_examples/plot_4_partial_dependencies.html>`_
These are very helpful to see what is happening during the optimization, and what can be adjusted
if necessary.
