.. _EVC system:

**************************
Experiment Version Control
**************************

Oríon comes with an Experiment Version Control (EVC) system that makes it possible to reuse results
from your previous experiments in a given project for the current one. This means a new experiment
could pre-train on all prior data resulting in a much more efficient optimization algorithm. Another
advantage of the EVC system is that it provides a systematic way to organize research and the
possibility to go back in time and compare the evolution of performance throughout your research.

Experiments inside the EVC are organized by version. When enabled (See :ref:`config_evc_enable`),
every time an experiment has
changed but has not been explicitly renamed, its version number will automatically increment and
this new version will appear as a new branch for that experiment.

However, it is possible to overrule the automatic resolution of changes for experiments by using the
`--manual-resolution` argument with Oríon. The rest of the document presents the process of doing
so.

To continue with our examples from pytorch-mnist, suppose we decide at some point we would like to
also optimize the ``momentum``. For the sake of brevity, the `--manual-resolution` argument has been
omitted from the command samples.

.. code-block:: bash

    $ orion hunt -n orion-tutorial python main.py --lr~'loguniform(1e-5, 1.0)' --momentum~'uniform(0, 1)'

This cannot be the same as the experiment ``orion-tutorial`` since the space of optimization is now
different. Such a call will trigger an experiment branching, meaning that a new experiment will
be created which points to the previous one, ``orion-tutorial``, the one without momentum in this
case.

.. code-block:: text

    Welcome to Orion's experiment branching interactive conflicts resolver
    -----------------------------------------------------------------------

    If you are unfamiliar with this process, you can type `help` to print the help message.
    You can also type `abort` or `(q)uit` at any moment to quit without saving.

    Remaining conflicts:

       Experiment name 'orion-tutorial' already exist for user 'bouthilx'
       New momentum

    (orion)

You should think of it like a ``git status``. It tells you want changed and what you did not commit
yet. If you hit tab twice, you will see all possible commands. You can enter `h` or `help` for more
information about each command. In this case we will first add ``momentum``. You can enter ``add``
and then hit tab twice. Oríon will detect any possible hyper-parameter that you could add and
autocomplete it. Since we only have ``momentum`` in this case, it will be fully autocompleted. If
you hit tab twice again, the option ``--default-value`` will be added to the line, with which you
can set a default-value for the momentum. If you only enter ``add momentum``, the new experiment
won't be able to fetch trials from the parent experiment, because it cannot know what was the
implicit value of ``momentum`` on those trials. If you know there was a default value
for ``momentum``, you should tell so with ``--default-value``.


.. code-block:: text

    (orion) add momentum --default-value 0
    TIP: You can use the '~+' marker in place of the usual ~ with the command-line to solve this
    conflict automatically.
    Ex: -x~+uniform(0,1)

    Resolutions:

         momentum~+uniform(0, 1, default_value=0.0)


    Remaining conflicts:

         Experiment name 'orion-tutorial' already exist for user 'bouthilx'

    (orion)

As you can see, when resolving the conflicts with the prompt, Oríon will always tell you how
you could have resolved the conflict directly in commandline. If we follow the advice, we would
change our commandline like this.

.. code-block:: bash

    $ orion hunt -n orion-tutorial python main.py --lr~'loguniform(1e-5, 1.0)' --momentum~+'uniform(0, 1)'

Let's look back at the prompt above. Following the resolution of ``momentum`` conflict we see
that it is now marked as resolved in the `Resolutions` list, while the experiment name is still
marked as a conflict. Notice that the prior distribution is slightly different than the one
specified in commandline. This is because we added a default value inside the prompt. Notice
also that the resolution is marked as how you would resolve this conflict in commandline.
There are hints everywhere to help you learn without looking at the documentation.

Now for the experiment name conflict. Remember that experiment names must be unique, that means that
when an experiment branching occur we need to give a new name to the child experiment. You can do so
with the command ``name``. If you hit tab twice with ``name``, Oríon will auto-complete with all
experiment names in the current project. This makes it easy to autocomplete an experiment name and
simply append some version number like ``1.2`` at the end. Let's add ``-with-momentum`` in our case.

.. code-block:: text

    (orion) add orion-tutorial-with-momentum
    TIP: You can use the '-b' or '--branch' command-line argument to automate the naming process.

    Resolutions:

         --branch orion-tutorial-with-momentum
         momentum~+uniform(0, 1, default_value=0.0)


    Hooray, there is no more conflicts!
    You can enter 'commit' to leave this prompt and register the new branch


    (orion)

Again Oríon will tell you how you can resolve an experiment name conflict in command-line to avoid
the prompt, and the resolution will be marked accordingly.

.. code-block:: bash

    $ orion hunt -n orion-tutorial -b orion-tutorial-with-momentum python main.py --lr~'loguniform(1e-5, 1.0)' --momentum~+'uniform(0, 1)'

You can execute again this branched experiment by reusing the same commandline but replacing the new
experiment name ``orion-tutorial-with-momentum``.

.. code-block:: bash

    $ orion hunt -n orion-tutorial-with-momentum python main.py --lr~'loguniform(1e-5, 1.0)' --momentum~'uniform(0, 1)'

Or as always by only specifying the experiment name.

.. code-block:: bash

    $ orion hunt -n orion-tutorial-with-momentum

If you are unhappy with some resolutions, you can type ``reset`` and hit tab twice. Oríon will
offer autocompletions of the possible resolutions to reset.

.. code-block:: text

    (orion) reset '
    '--branch orion-tutorial-with-momentum'
    'momentum~+uniform(0, 1, default_value=0.0)'
    (orion) reset '--branch orion-tutorial-with-momentum'

    Resolutions:

         momentum~+uniform(0, 1, default_value=0.0)


    Remaining conflicts:

         Experiment name 'orion-tutorial' already exist for user 'bouthilx'

    (orion)

Once you are done, you can enter ``commit`` and the branched experiment will be register and will
begin execution.

Source of conflicts
-------------------

1. Code modification
2. Commandline modification
3. Script configuration file modification
4. Optimization space modification (new hyper-parameters or change of prior distribution)
5. Algorithm configuration modification
6. Orion version change

Iterative Results
=================

.. note:: TODO: Set link when status command is documented.

Results from the EVC tree can be queried in aggregation with the command
`status --collapse <idontexist>`_ or fetching using the
:ref:`library API <library-api-evc-results>`.
