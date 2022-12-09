*****************************************************************************************
Hyperparameters optimisation using a HuggingFace Model and the Hydra-Orion-Sweeper plugin
*****************************************************************************************

In this tutorial, we will show an easy Orion integration of a HuggingFace translation model using Hydra, with the 
`Hydra_Orion_Sweeper <https://github.com/Epistimio/hydra_orion_sweeper >`_ plugin. Hydra is essentially a framework for
configuring applications. We will use it to define our Hyperparameters and some Orion configuration. We will also be
using `Comet <https://www.comet.com/>`_ for monitoring our experiments.

Imports
^^^^^^^
You will need to import some modules using pip, such as ``datasets``, ``transformers``, ``hydra-orion-sweeper`` and ``comet-ml``.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 5-6,13-22


Hydra configuration file
^^^^^^^^^^^^^^^^^^^^^^^^

Notice here how the arguments that are not defined will be set at ``None``, which will be overridden by default values
or not used at all. This serves as a replacement for parsing arguments in the command line, but is integrated with Orion,
which makes the usability really good.

.. literalinclude:: /../../examples/huggingface/config.yaml
  :language: yaml
  :lines: 1-

If you want to change your working or/and logging directory, you can also easily do that.
From the config file, you can specify 
 
.. literalinclude:: /../../examples/huggingface/config.yaml
  :language: yaml
  :lines: 29-31

This will change your working directory. You can see that with the hydra-orion-sweeper, we are able to specify 4 variables from Orion.
``${hydra.sweeper.orion.name}`` for the ``experiment_name``, ``${hydra.sweeper.orion.id}`` for the ``experiment_id``, 
``${hydra.sweeper.orion.uuid}`` for the experiment ``uuid`` and ``${hydra.sweeper.orion.trial}`` for the ``trial_id``.

In the code, you can now specify the output directory to the trainer with the ``output_dir`` parameter.
``os.getcwd()`` specifies the current working dir.

.. code-block:: python
    output_dir=str(os.getcwd())+"/test_trainer", 

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
  :lines: 63-68, 229-230

With the ``hydra-orion-sweeper``, this function needs to return the objective. You have 3 choices 
for what you can return : 

.. code-block:: python

    if cfg.return_type == "float":
        return out

    if cfg.return_type == "dict":
        return dict(name="objective", type="objective", value=out)

    if cfg.return_type == "list":
        return [dict(name="objective", type="objective", value=out)]

For the purpose of this tutorial, we are going to keep it simple and simply return a float, our objective that we want to minimize.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 224

Comet
^^^^^
We are going to use Comet to track our experiments. It is quite simple to use. First, install comet using 

.. code-block:: bash

   $ pip install comet-ml

Now that it is installed, we simply have to set some environment variables, such as 
 
.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 7-10

You can also set them in your working environment. If you are to set them in python, however, you need to make sure to set them before
importing ``transformers``.

For the ``COMET_API_KEY``, you will be given a token when creating your comet account. This is the one you are going to use here.

And that is it ! If the variables are set and comet-ml is downloaded, HuggingFace will Automatically upload your data to Comet, you simply have to 
go to your profile on their site and see your experiments.

Example code
^^^^^^^^^^^^
For this example, we are fine-tuning a pretrained translation model named ``Helsinki-NLP``. We start by setting the training arguments.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 90-98
  
For our purposes, we will be using a ``Seq2SeqTrainer``, so for the training arguments are going to be ``Seq2SeqTrainingArguments``.
The ``set_training_args`` function adds the hydra arguments into the training arguments.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 43-47

For the dataset, we are going to use the ``wmt16`` dataset. We can set a ``cache_dir`` as to where the dataset cache will be stored

.. literalinclude:: /../../examples/huggingface/main.py
   :language: python
   :lines: 103,106
   
We then prepare our training and evaluation datasets. In this example, we want to evaluate our model with the validation dataset 
and the training dataset. 

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 128-152

For the metric, we are going to use ``sacrebleu``. We can also set a ``cache_dir`` here for the metric cache files. The ``compute_metrics``
function goes as follow :

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 162-163, 164-188

Now we have to create the actual Trainer, a ``Seq2SeqTrainer`` as mentioned previously. It is very much like a classic ``Trainer`` from HuggingFace.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 204-212

Automatically, HuggingFace will log the evaluation (to Comet) from the ``eval_dataset``. Since we also want the evaluation from the 
training dataset, we will have to implement something called a ``CustomCallback``. The one I made for this tutorial takes the ``trainer``
and the dataset we want to add (In our case, our train dataset) as parameters. We can then rewrite some callback functions, such as ``on_epoch_end()``

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 190-201,214

All that is left to do now is to train the model, and once it's finish training, send the data to Orion by returning it.

.. literalinclude:: /../../examples/huggingface/main.py
  :language: python
  :lines: 215-217, 224

For more details, feel free to simply go look at the code, in ``examples/huggingface/main.py``

Execution
^^^^^^^^^
We simply have to run the main.py file with the -m argument, which makes sure we use the Hydra-Orion-Sweeper plugin.

.. code-block:: bash

   $ python3 main.py -m