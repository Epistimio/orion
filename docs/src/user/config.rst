.. _configuration:

**********************
Advanced Configuration
**********************

Oríon supports different levels of configuration to provide flexibility.
The configuration levels are hierchical with higher levels having precedence over the
lower ones. The levels are the following:

1. Default configuration
2. Global configuration
3. Environment variables
4. Experiment configuration from database*
5. Local configuration
6. Commandline arguments

Where larger numbers have precedence over
the lower ones. We describe here further each
type of configuration.

**1. Default Configuration**

Default values defined in core code of Oríon  `here </_modules/orion/core.html>`_.

**2. Global Configuration**

Defined in a yaml file in global configuration folders of Oríon.
On Linux systems, this is typically at ``$HOME/.config/orion.core``. You can get a list
of these folders on your system with the following command.

.. code-block:: bash

   $ python -c 'import orion.core; print("\n".join(orion.core.DEF_CONFIG_FILES_PATHS))'
   /usr/share/awesome/orion.core/orion_config.yaml.example
   /etc/xdg/xdg-awesome/orion.core/orion_config.yaml
   /home/user/.config/orion.core/orion_config.yaml

This list is an example and will likely be differ on your end.

**3. Environment Variables**

Most options are configurable through environment variables as well.
These are provided in the documented below at the **Env var** fields,
whenever configurable otherwise the field is left empty.

**4. Experiment Configuration from Database**

\*Experiment configuration from database is not configurable by the user per say,
but it is represented here because whenever something differs between the experiment's
configuration and the levels below, the experiment's configuration will have precedence.
For instance if ``experiment.max_trials`` is set to 10 in global configuration
and an experiment with ``max_trials`` set to 100 is resumed, then ``max_trials``
will be 100, not 10.
However, if ``experiment.max_trials`` is set to 10 in local configuration file
(the file passed with ``--config`` to the hunt command) or with commandline argument
``--exp-max-trials``, then the experiments ``max_trials`` will be updated to 10.
This make it possible to resume experiments without specifying the whole configuration,
because experiment configuration from database is reused, but it also makes it possible
to create a new experiment based on a previous one by simply specifying what to change.

**5. Local Configuration**

Defined in a yaml file that is passed to commandline.

.. code-block:: bash

      orion [COMMAND] --config local.yaml

Not to be confused with a configuration file that may be passed to the user's script.

.. code-block:: bash

      orion hunt --config local.yaml ./myscript.sh --config script.yaml

Here ``script.yaml`` is the user's script configuration (passed to ``./myscript.sh``),
and ``local.yaml`` is the local configuration script (passed to ``hunt``).

The configuration passed through the python API is also considered as local configuration.

**6. Commandline Arguments**

All arguments of ``orion hunt``.

Full Example of Global Configuration
------------------------------------

.. code-block:: yaml

    storage:
      database:
        host: orion_db.pkl
        type: pickleddb

    experiment:
        algorithm:
            random:
                seed: None
        max_broken: 3
        max_trials: 1000000000
        worker_trials: 1000000000
        working_dir:

    worker:
        n_workers: 1
        pool_size: 0
        executor: PoolExecutor
        executor_configuration: {}
        heartbeat: 120
        interrupt_signal_code: 130
        max_broken: 10
        idle_timeout: 60
        max_trials: 1000000000
        user_script_config: config

    evc:
        algorithm_change: False
        auto_resolution: True
        cli_change_type: break
        code_change_type: break
        config_change_type: break
        orion_version_change: False
        ignore_code_changes: False
        manual_resolution: False
        non_monitored_arguments: []


----


.. _config_database:

Database
--------

.. code-block:: yaml

    database:
        host: orion_db.pkl
        type: pickleddb


.. _config_database_name:

name
~~~~

:Type: str
:Default: orion
:Env var: ORION_DB_NAME
:Description:
    Name of the database.



.. _config_database_type:

type
~~~~

:Type: str
:Default: PickledDB
:Env var: ORION_DB_TYPE
:Description:
    Type of database. Builtin backends are ``mongodb``, ``pickleddb`` and ``ephemeraldb``.



.. _config_database_host:

host
~~~~

:Type: str
:Default: ""
:Env var: ORION_DB_ADDRESS
:Description:
    URI for ``mongodb``, or file path for ``pickleddb``.  An empty
    string will set the value depending on the database type
    (localhost or <user_data_dir>/orion/orion_db.pkl).



.. _config_database_port:

port
~~~~

:Type: int
:Default: 27017
:Env var: ORION_DB_PORT
:Description:
    Port address for ``mongodb``.



----


.. _config_experiment:

Experiment
----------

.. code-block:: yaml

    experiment:
        algorithm:
            random:
                seed: None
        max_broken: 3
        max_trials: 1000000000
        worker_trials: 1000000000
        working_dir:



.. _config_experiment_name:

name
~~~~

.. note:: This option is only supported in local configuration.

:Type: str
:Default:
:Env var:
:Description:
    Name of the experiment.


.. _config_experiment_version:

version
~~~~~~~

.. note:: This option is only supported in local configuration.


:Type: int
:Default: None
:Env var:
:Description:
    Version of the experiment. If not defined, latest experiment for the given
    name will be selected. Version is automatically incremented if there is any
    modification detected in the experiment's configuration
    (search space, algorithm configuration, code version, ...)


user
~~~~

.. note:: This option is only supported in local configuration.

:Type: str
:Default: $USERNAME
:Env var:
:Description:
    Name of the user to associate with the experiment.


.. _config_experiment_max_trials:

max_trials
~~~~~~~~~~

:Type: int
:Default: 1000000000
:Env var: ORION_EXP_MAX_TRIALS
:Description:
    number of trials to be completed for the experiment. This value will be saved within the
    experiment configuration and reused across all workers to determine experiment's completion.



.. _config_experiment_worker_trials:

worker_trials
~~~~~~~~~~~~~

.. warning::

   **DEPRECATED.** This argument will be removed in v0.3.
   See :ref:`worker: max_trials <config_worker_max_trials>` instead.

:Type: int
:Default: 1000000000
:Env var:
:Description:
    (DEPRECATED) This argument will be removed in v0.3.
    See :ref:`worker: max_trials <config_worker_max_trials>` instead.



.. _config_experiment_max_broken:

max_broken
~~~~~~~~~~

:Type: int
:Default: 3
:Env var: ORION_EXP_MAX_BROKEN
:Description:
    Maximum number of broken trials before experiment stops.



.. _config_experiment_working_dir:

working_dir
~~~~~~~~~~~

:Type: str
:Default:
:Env var: ORION_WORKING_DIR
:Description:
    Set working directory for running experiment.



.. _config_experiment_algorithms:

algorithm
~~~~~~~~~~

:Type: dict
:Default: random
:Env var:
:Description:
    Algorithm configuration for the experiment.



.. _config_experiment_strategy:

strategy
~~~~~~~~

.. warning::

   **DEPRECATED.** This argument will be removed in v0.4.
   Set parallel strategy in algorithm configuration directly, if the algorithm supports it.

:Type: dict
:Default: MaxParallelStrategy
:Env var:
:Description:
    (DEPRECATED) This argument will be removed in v0.4. Parallel strategies are now handled by
    algorithm directly and should be set in algorithm configuration when they support it.

----


.. _config_worker:

Worker
------

.. code-block:: yaml

    worker:
        n_workers: 1
        pool_size: 0
        executor: PoolExecutor
        executor_configuration: {}
        heartbeat: 120
        interrupt_signal_code: 130
        max_broken: 10
        idle_timeout: 60
        max_trials: 1000000000
        user_script_config: config



.. _config_worker_n_workers:

n_workers
~~~~~~~~~

:Type: int
:Default: 1
:Env var: ORION_N_WORKERS
:Description:
    Number of workers to run in parallel.
    It is possible to run many `orion hunt` in parallel, and each will spawn
    ``n_workers``.

.. _config_worker_pool_size:

pool_size
~~~~~~~~~

:Type: int
:Default: 0
:Env var:
:Description:
    Number of trials to sample at a time. If 0, default to number of workers.
    Increase it to improve the sampling speed if workers spend too much time
    waiting for algorithms to sample points. An algorithm will try sampling `pool_size`
    trials but may return less.


.. _config_worker_executor:

executor
~~~~~~~~

:Type: str
:Default: PoolExecutor
:Env var: ORION_EXECUTOR
:Description:
    The executor backend used to parallelize orion workers.


.. _config_worker_executor_configuration:

executor_configuration
~~~~~~~~~~~~~~~~~~~~~~

:Type: str
:Default: {}
:Description:
    The configuration of the executor. See :py:mod:`orion.executor` for documentation
    of executors configuration.


.. _config_worker_heartbeat:

heartbeat
~~~~~~~~~

:Type: int
:Default: 120
:Env var: ORION_HEARTBEAT
:Description:
    Frequency (seconds) at which the heartbeat of the trial is updated. If the heartbeat of a
    `reserved` trial is larger than twice the configured heartbeat, Oríon will reset the status of
    the trial to `interrupted`. This allows restoring lost trials (ex: due to killed worker).



.. _config_worker_max_trials:

max_trials
~~~~~~~~~~

:Type: int
:Default: 1000000000
:Env var: ORION_WORKER_MAX_TRIALS
:Description:
    Number of trials to be completed for this worker. If the experiment is completed, the worker
    will die even if it did not reach its maximum number of trials.



.. _config_worker_max_broken:

max_broken
~~~~~~~~~~

:Type: int
:Default: 3
:Env var: ORION_WORKER_MAX_BROKEN
:Description:
    Maximum number of broken trials before worker stops.


.. _config_worker_idle_timeout:

idle_timeout
~~~~~~~~~~~~~~~~~~~

:Type: int
:Default: 60
:Env var: ORION_IDLE_TIMEOUT
:Description:
    Maximum time the experiment can spend trying to reserve a new suggestion. Such timeout are
    generally caused by slow database, large number of concurrent workers leading to many race
    conditions or small search spaces with integer/categorical dimensions that may be fully
    explored.


.. _config_worker_max_idle_time:

max_idle_time
~~~~~~~~~~~~~

.. warning::

   **DEPRECATED.** This argument will be removed in v0.3.
   Use :ref:`config_worker_idle_timeout` instead.

:Type: int
:Default: 60
:Env var: ORION_MAX_IDLE_TIME
:Description:
    (DEPRECATED) This argument will be removed in v0.3. Use :ref:`config_worker_idle_timeout`
    instead.


.. _config_worker_interrupt_signal_code:

interrupt_signal_code
~~~~~~~~~~~~~~~~~~~~~

:Type: int
:Default: 130
:Env var: ORION_INTERRUPT_CODE
:Description:
    Signal returned by user script to signal to Oríon that it was interrupted.



.. _config_worker_user_script_config:

user_script_config
~~~~~~~~~~~~~~~~~~

:Type: str
:Default: config
:Env var: ORION_USER_SCRIPT_CONFIG
:Description:
    Config argument name of user's script (--config).



----


.. _config_evc:

Experiment Version Control
--------------------------

.. code-block:: yaml

    evc:
        enable: False
        algorithm_change: False
        auto_resolution: True
        cli_change_type: break
        code_change_type: break
        config_change_type: break
        orion_version_change: False
        ignore_code_changes: False
        manual_resolution: False
        non_monitored_arguments: []


.. _config_evc_enable:

enable
~~~~~~~~~~~~~~~

.. note::

   New in version v0.1.16. Previously the EVC was always enabled. It is now disable by default
   and can be enabled using this option.

:Type: bool
:Default: False
:Env var: ORION_EVC_ENABLE
:Description:
    Enable the Experiment Version Control. Defaults to False. When disabled, running
    an experiment different from an earlier one but sharing the same name will have the
    effect of overwriting the previous one in the database. Trials of the previous experiment
    will still point to the experiment but may be incoherent with the new search space.


.. _config_evc_auto_resolution:

auto_resolution
~~~~~~~~~~~~~~~

.. warning::

   **DEPRECATED.** This argument will be removed in v0.3.
   See :ref:`evc: manual_resolution <config_evc_manual_resolution>` to avoid auto-resolution.

:Type: bool
:Default: True
:Env var:
:Description:
    (DEPRECATED) This argument will be removed in v0.3. Conflicts are now resolved automatically by
    default. See :ref:`evc: manual_resolution <config_evc_manual_resolution>` to avoid
    auto-resolution.


.. _config_evc_manual_resolution:

manual_resolution
~~~~~~~~~~~~~~~~~

:Type: bool
:Default: False
:Env var: ORION_EVC_MANUAL_RESOLUTION
:Description:
    If ``True``, enter experiment version control conflict resolver for manual resolution on
    branching events. Otherwise, auto-resolution is attempted.



.. _config_evc_non_monitored_arguments:

non_monitored_arguments
~~~~~~~~~~~~~~~~~~~~~~~

:Type: list
:Default: []
:Env var: ORION_EVC_NON_MONITORED_ARGUMENTS
:Description:
    Ignore these commandline arguments when looking for differences in user's commandline call.
    Environment variable and commandline only supports one argument. Use global config or local
    config to pass a list of arguments to ignore. When defined with environment variables,
    use ':' to delimit multiple arguments (ex: 'three:different:arguments').



.. _config_evc_ignore_code_changes:

ignore_code_changes
~~~~~~~~~~~~~~~~~~~

:Type: bool
:Default: False
:Env var: ORION_EVC_IGNORE_CODE_CHANGES
:Description:
    If ``True``, ignore code changes when looking for differences.



.. _config_evc_algorithm_change:

algorithm_change
~~~~~~~~~~~~~~~~

:Type: bool
:Default: False
:Env var: ORION_EVC_ALGO_CHANGE
:Description:
    If ``True``, set algorithm change as resolved if a branching event occur. Child and parent
    experiment have access to all trials from each other when the only difference between them is
    the algorithm configuration.


.. _config_evc_code_change_type:

code_change_type
~~~~~~~~~~~~~~~~

:Type: str
:Default: break
:Env var: ORION_EVC_CODE_CHANGE
:Description:
    One of ``break``, ``unsure`` or ``noeffect``. Defines how trials should be filtered in
    Experiment Version Control tree if there is a change in the user's code repository. If the
    effect of the change is ``unsure``, the child experiment will access the trials of the parent
    but not the other way around. This is to ensure parent experiment does not get corrupted with
    possibly incompatible results. The child cannot access the trials from parent if
    ``code_change_type`` is ``break``. The parent cannot access trials from child if
    ``code_change_type`` is ``unsure`` or ``break``.



.. _config_evc_cli_change_type:

cli_change_type
~~~~~~~~~~~~~~~

:Type: str
:Default: break
:Env var: ORION_EVC_CMDLINE_CHANGE
:Description:
    One of ``break``, ``unsure`` or ``noeffect``. Defines how trials should be filtered in
    Experiment Version Control tree if there is a change in the user's commandline call. If the
    effect of the change is ``unsure``, the child experiment will access the trials of the parent
    but not the other way around. This is to ensure parent experiment does not get corrupted with
    possibly incompatible results. The child cannot access the trials from parent if
    ``cli_change_type`` is ``break``. The parent cannot access trials from child if
    ``cli_change_type`` is ``unsure`` or ``break``.



.. _config_evc_config_change_type:

config_change_type
~~~~~~~~~~~~~~~~~~

:Type: str
:Default: break
:Env var: ORION_EVC_CONFIG_CHANGE
:Description:
    One of ``break``, ``unsure`` or ``noeffect``. Defines how trials should be filtered in
    Experiment Version Control tree if there is a change in the user's script. If the effect of the
    change is ``unsure``, the child experiment will access the trials of the parent but not the
    other way around. This is to ensure parent experiment does not get corrupted with possibly
    incompatible results. The child cannot access the trials from parent if ``config_change_type``
    is ``break``.  The parent cannot access trials from child if ``config_change_type`` is
    ``unsure`` or ``break``.


.. _config_evc_orion_version_change:

orion_version_change
~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Default: False
:Env var: ORION_EVC_ORION_VERSION_CHANGE
:Description:
    If ``True``, set orion version change as resolved if branching event occurred.
    Child and parent experiment have access to all trials from each other
    when the only difference between them is the orion version used during execution.
