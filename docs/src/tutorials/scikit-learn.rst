********************
Scikit-learn
********************
.. Might also think of moving this file to examples/ then we an example auto contained in the
   repository. We invoke this file from index.rst

In this tutorial, we're going to demonstrate how Oríon can be integrated to a minimal model using
`scikit-learn <https://scikit-learn.org/stable/index.html>`_ on the `iris dataset
<http://archive.ics.uci.edu/ml/datasets/Iris>`_. The files mentioned in this tutorial are available
at `examples/scikitlearn-iris/
<https://github.com/Epistimio/orion/tree/master/examples/scikitlearn-iris/>`_ in Oríon's repository.

The requirements are listed in requirements.txt. You can quickly install them using ``$ pip
install -r requirements.txt``. If you haven't installed Oríon previously, make sure to
:doc:`configure it properly </install/core>` before going further.

Sample script
---------------

.. literalinclude:: /../../examples/scikitlearn-iris/main.py
   :language: python
   :lines: 1-9, 13-30

This very basic script takes in parameter one positional argument for the hyper-parameter *epsilon*
which control the loss in the script.

The script is divided in five parts:

#. Parsing of the script arguments
#. Loading and splitting the dataset
#. Training a classifier using the researcher-defined *epsilon*
#. Evaluating the classifier using the testing set
#. Reporting the performance of the model. i.e., the accuracy.

.. note::
   The workflow presented in the script is simplified on purpose compared to real ones. The
   objective of this example is to illustrate the basic steps involved in using Oríon.

To find a good *epsilon*, a user would run empirically ``$ python main.py <epsilon>`` multiple
times, choosing a new value for *epsilon* manually.

This ad-hoc hyper-parameter optimization is unreliable, slow, and requires a lot of work from the
user. Oríon solves this problem by providing established hyper-parameter optimization
algorithms without disrupting the workflow of the user. Integrating it only require minimal
adjustments to your current workflow as we'll demonstrate in the next section.

Enter Orion
-----------
Integrating Oríon into your workflow requires only two non-invasive changes:
   1. Define an objective to optimize.
   2. Specify the hyper-parameter space.

For the former, this step takes place in the script training the model. The latter can either be
specified in a configuration file or directly while calling the script with Oríon.
For the purpose of the example, we'll configure the hyper-parameter space directly as a
command-line argument.

Updating the script
^^^^^^^^^^^^^^^^^^^
We only need to make one small change to the script: we report to Oríon the objective that we
want to **minimize** at the end of the script using :py:func:`orion.client.report_objective`:

.. literalinclude:: /../../examples/scikitlearn-iris/main.py
   :language: python
   :lines: 32-

In our example, we measure the accuracy of the model to qualify its performance. To get the best
accuracy possible, we need to minimize the difference between 1 and the accuracy to get it as
close to 1 as possible. Otherwise, we'll be minimizing the accuracy which will yield a poor model.

:py:func:`orion.client.report_objective` can be imported using :

.. code-block:: python

   from orion.client import report_objective

Updating the script call
^^^^^^^^^^^^^^^^^^^^^^^^
The last missing piece in automating the hyper-parameter optimization of our example model is to
supply Oríon with the values to use for *epsilon*.

We specify the search space in the command line using ``orion~loguniform(1e-5, 1.0)``
as the argument for *espilon*. This argument will tell Oríon to use a log uniform distribution
between ``1e-5`` and ``1`` for the values of *epsilon*.

Putting everything together, we need to call ``main.py`` with Oríon. The syntax is the
following: ``$ orion hunt python main.py 'orion~loguniform(1e-5, 1.0)'``. Before executing it on
your terminal, you have to specify the name of the experiment using the ``-n`` option. It is also a
good idea to specify a stopping condition using ``--max-trials`` otherwise the optimization will
not stop unless you interrupt it with :kbd:`ctrl-c`:

.. code-block:: bash

   $ orion hunt -n scitkit-iris-tutorial --max-trials 50 python main.py 'orion~loguniform(1e-5, 1.0)'

.. warning::
   Make sure you installed the dependencies for the script before running it using ``pip install
   -r requirements.txt``.

Viewing the results
-------------------
Once the optimization reached its stopping condition, you can query Oríon to give you the results
of the optimization with the sub-command ``$ orion info``:

.. code-block:: bash

   $ orion info -n scitkit-iris-tutorial

You can also query the results from the database using :ref:`Oríon's python API
<library-api-results>`. Check it out to learn more and see examples.
