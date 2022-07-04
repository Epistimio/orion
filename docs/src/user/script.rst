*********
Integrate
*********

This section describes how to adapt a user script with Oríon.
To customize how Oríon parses the commandline or execution environment see :ref:`customization`.
If the user script requires information about the running trial, such as its id,
the working directory or the experiment's name, look at :ref:`commandline_templates` or
:ref:`env_vars`.

.. _customization:

Orion customization
===================

``user_script_config``
----------------------

By default Oríon will only consider the file passed through the argument ``--config`` as a
configuration file for the user script. However, it is possible to change the default argument
inside the global configuration file of Oríon through the ``user_script_config`` argument, like
this:

.. code-block:: yaml

    worker:
        user_script_config: configuration

It is then possible to run ``orion hunt`` like here:

.. code-block:: console

   orion hunt --config my_orion_config.yaml ./my_script --configuration my_script_config.txt

As you can see, the configuration file for the user script is now passed through
``--configuration``.
This should not be confused with the argument ``--config`` of ``orion hunt``,
which is the configuration of Oríon. We are here referring the configuration of the user script,
represented with ``my_script_config.txt`` in the previous example.


``working_dir``
---------------

By default the working directory where trial configuration scripts are saved will be a temporary
folder that is deleted at end of execution of the trial. To keep the data in a particular folder,
you can specify ``--working-dir`` or define it in the configuration file, either the global one or
the local one passed to
``hunt --config <file_path>``. This working directory is where all trial directories will be
created. To access the particular working directory of a trial, see next sections
:ref:`commandline_templates` and :ref:`env_vars`.

.. _language_compatibility:

Language compatibility
======================
The command line works for scripts and programs in any language.
The only requirement is that the executed script returns a JSON string with the objective value.

The format is

.. code-block:: json

   [
      {
         "name": "some-objective",
         "type": "objective",
         "value": 1
      }
   ]


See :meth:`orion.client.report_results` for more details.

.. _commandline_templates:

Command-line templating
=======================

When parsing the commandline of the experiment to execute a trial, Oríon will replace any
templates referencing the experiment or the trial. This means that execution of the following line

.. code-block:: console

   orion hunt [...] ./my_script [...] --dir '{trial.working_dir}'


would replace ``{trial.working_dir}`` with the actual path of the trial's working dir. Since it is
using templating, you can place the templates wherever you want including formats such as

.. code-block:: console

   orion hunt [...] ./my_script [...] --some-arg some-weird-{exp.name}-{trial.id}-value


Here is a list of convenient attributes of the ``Experiment`` and ``Trial`` objects which
are available through templates.

========================== ====================================
Templates                  Description
========================== ====================================
``exp.id``                 ID of the experiment

``exp.name``               Name of the experiment

``exp.version``            Version of the experiment

``exp.working_dir``        Global working dir of the experiment

``trial.id``               Unique ID of the trial

``trial.legacy_id``        Deprecated. Unique ID of the trial prior to v0.2.5.

``trial.working_dir``      Working dir of the trial

``trial.hash_params``      md5sum hash for the parameters (w/o fidelity)

========================== ====================================

.. note::

   Templates are only supported in commandline and not in user script configuration
   at the moment.  We plan to support both in the future. Contributions are very welcome. :)

.. _env_vars:

Environment variables
=====================

When executing the trial, Oríon will set local environment variables available to the user script.
We list them below.

.. envvar:: ORION_EXPERIMENT_ID

   Current experiment that is being ran.

.. envvar::  ORION_EXPERIMENT_NAME

   Name of the experiment the worker is currently working on.

.. envvar::  ORION_EXPERIMENT_VERSION

   Version of the experiment the worker is currently working on.

.. envvar:: ORION_TRIAL_ID

   Current trial id that is currently being executed in this process.

.. envvar:: ORION_TRIAL_LECAGY_ID

   Deprecated. Unique ID of the trial prior to v0.2.5.

.. envvar:: ORION_WORKING_DIR

   Trial's current working directory.

.. envvar:: ORION_RESULTS_PATH

   Trial's results file that is read by the legacy protocol to get the results of the trial
   after a successful run.
