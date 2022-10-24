.. role:: hidden
    :class: hidden-section


.. _storage:

*******
Storage
*******

Commands are available to help
:ref:`configure <storage_setup>`,
:ref:`test <storage_test>` and
:ref:`upgrade <storage_upgrade>` the storage of Oríon.
There is additionally commands to :ref:`delete <storage_rm>` experiment and trials
or :ref:`update <storage_set>` values in the storage.

For more flexibility, there is the :ref:`storage_python_apis`.

.. _storage_commands:

Commands
========

.. _storage_setup:

``setup`` Storage configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``setup`` command helps creating a global configuration file
for the configuration of Oríon's storage. For more details on its usage
see :ref:`Database Configuration` in the database
installation and configuration section.

.. _storage_test:

``test`` Test storage configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``test`` command provides a simple and efficient way of testing the storage configuration. For
more details on its usage see :ref:`Test Connection` in the database installation and configuration
section.

.. _storage_rm:

``rm`` Delete data from storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Command to delete experiments and trials.

To delete an experiment and its trials, simply give the experiment's name.

.. code-block:: sh

   orion db rm my-exp-name

To delete only trials that are broken, simply add ``--status`` broken.
Note that the experiment will not be deleted, only the trials.

.. code-block:: sh

   orion db rm my-exp-name --status broken

Or ``--status *`` to delete all trials of the experiment.

.. code-block:: sh

   orion db rm my-exp-name --status *

By default, the last version of the experiment is deleted. Add ``--version``
to select a prior version. Note that all child of the selected version
will be deleted as well. You cannot delete a parent experiment without
deleting the child experiments.

.. code-block:: sh

   orion db rm my-exp-name --version 1


.. _storage_set:

``set`` Change value of data in storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Command to update trial attributes.

To change a trial status, simply give the experiment name,
trial id and status. (use `orion status --all` to get trial ids)

.. code-block:: sh

   orion db set my-exp-name id=3cc91e851e13281ca2152c19d888e937 status=interrupted

To change all trials from a given status to another, simply give the two status

.. code-block:: sh

   orion db set my-exp-name status=broken status=interrupted

Or `*` to apply the change to all trials

.. code-block:: sh

   orion db set my-exp-name '*' status=interrupted

By default, trials of the last version of the experiment are selected.
Add --version to select a prior version. Note that the modification
is applied recursively to all child experiment, but not to the parents.

.. code-block:: sh

   orion db set my-exp-name --version 1 status=broken status=interrupted


.. _storage_release:

``release`` algorithm lock
~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm state is saved in the storage so that it can be shared across main process
(``$ orion hunt`` or ``experiment_client.workon()``). The algorithm state is locked
during the time the algorithm is updated by observing completed trials or during the
suggestion of new trials. Sometimes the process may be killed while the algorithm is locked
leading to a dead lock. The lock can be manually released using the ``orion db release``.

.. code-block:: sh

   orion db release my-exp-name --version 1

Make sure you have no Orion process running with this experiment while executing this command
or you risk having an algorithm state saved in the storage that is inconsistent with the trials
saved in the storage.

.. _storage_upgrade:

``upgrade`` Upgrade database scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Database scheme may change from one version of Oríon to another. If such change happens, you will
get the following error after upgrading Oríon.

.. code-block:: sh

   The database is outdated. You can upgrade it with the command `orion db upgrade`.

Make sure to create a backup of your database before upgrading it. You should also make sure that no
process writes to the database during the upgrade otherwise the latter could fail. When ready,
simply run the upgrade command.

.. code-block:: sh

   orion db upgrade

.. _storage_python_apis:

Python APIs
===========

In short, users are expected to only use the
:py:class:`ExperimentClient <orion.client.experiment.ExperimentClient>` to interact
with the storage client, to fetch and register trials. Creation of experiments
should always be done through
:py:func:`create_experiment() <orion.client.create_experiment>`.

If you need to access the storage with more flexibility, you can do
so using the methods of the storage client directly. See :ref:`storage_backend` section
for more details.

Finally, legacy databases supported by Oríon can also be accessed directly in last
resort if the storage backend is not flexible enough. See :ref:`database_backend` section
for more details.



.. _experiment_client:

ExperimentClient
~~~~~~~~~~~~~~~~

The experiment client must be created with the helper function
:py:func:`get_experiment() <orion.client.get_experiment>` which will take care of
initiating the storage backend and load the corresponding experiment from the storage.
To create a new experiment use :py:func:`create_experiment() <orion.client.create_experiment>`.

There is a small subset of methods to fetch trials or register new ones. We focus here
on the methods for loading or creation of trials in particular, see
:py:class:`ExperimentClient <orion.client.experiment.ExperimentClient>` for documentation
of all methods.

The experiment client can be loaded in read-only or read/write mode. Make sure to
load the experiment with the proper mode if you want to edit the database.
For full read/write/execution rights, use
:py:func:`create_experiment() <orion.client.create_experiment>`.

Here is a short example to fetch trials or insert a new one.

.. code-block:: python

   from orion.client import create_experiment

   # Create the ExperimentClient
   experiment = create_experiment('exp-name', space=dict(x='uniform(0, 1)'))

   # To fetch all trials from an experiment
   trials = experiment.fetch_trials()

   # To fetch trials in a form on panda dataframe
   df = experiment.to_pandas()

   # Insert a new trial in storage
   experiment.insert(dict(x=0.5))

   # Insert a new trial and reserve to execute
   trial = experiment.insert(dict(x=0.6), reserve=True)

:hidden:`to_pandas`
----------------------

.. automethod:: orion.client.experiment.ExperimentClient.to_pandas
   :noindex:

:hidden:`fetch_trials`
----------------------

.. automethod:: orion.client.experiment.ExperimentClient.fetch_trials
   :noindex:

:hidden:`fetch_trials_by_status`
--------------------------------

.. automethod:: orion.client.experiment.ExperimentClient.fetch_trials_by_status
   :noindex:

:hidden:`fetch_noncompleted_trials`
-----------------------------------

.. automethod:: orion.client.experiment.ExperimentClient.fetch_noncompleted_trials
   :noindex:

:hidden:`get_trial`
-------------------

.. automethod:: orion.client.experiment.ExperimentClient.get_trial
   :noindex:

:hidden:`insert`
----------------

.. automethod:: orion.client.experiment.ExperimentClient.insert
   :noindex:



.. _storage_backend:

Storage
~~~~~~~

.. warning::

   The storage backends are not meant to be used directly by users.
   Be careful if you use any method which modifies the data in storage or
   you may break your experiment or trials.

The storage backend is used by the
:py:class:`ExperimentClient <orion.client.experiment.ExperimentClient>`
to read and write persistent records of the experiment and trials.
Although we recommend using the experiment client,
we document the storage backend here for users who may need
more flexibility.

There is two ways for creating the storage client. If you
already created an experiment client, the storage
was already created during the process of creating the
experiment client and you can get it with
:py:func:`orion.storage.base.get_storage`.
Otherwise, you can create the storage client with
:py:func:`orion.storage.base.setup_storage` before
fetching it with
:py:func:`get_storage() <orion.storage.base.get_storage>`.
To recap, you can create it indirectly with
:py:func:`create_experiment() <orion.client.create_experiment>`
or directly with
:py:func:`setup_storage() <orion.storage.base.setup_storage>`.
In both case, you can access it with
:py:func:`get_storage() <orion.storage.base.get_storage>`.

.. code-block:: python

   from orion.client import create_experiment
   from orion.storage.base import get_storage, setup_storage

   # Create the ExperimentClient and storage implicitly
   experiment = create_experiment('exp-name', space=dict(x='uniform(0, 1)'))

   # Or create storage explicitly using setup_storage
   setup_storage(dict(
       type='legacy',
       database=dict(
           type='pickleddb',
           host='db.pkl')
           )
       )
   )

   # Get the storage client
   storage = get_storage()

   # fetch trials
   trials = storage.fetch_trials(uid=experiment.id)

   # Update trial status
   storage.set_trial_status(trials[0], 'interrupted')

.. note::

   The function :py:func:`setup_storage() <orion.storage.base.setup_storage>`
   reads the global configuration like
   :py:func:`create_experiment() <orion.client.create_experiment>`
   does if there is missing information. Therefore, it is possible
   to call it without any argument the same way it is possible
   to call
   :py:func:`create_experiment() <orion.client.create_experiment>`
   without specifying storage configuration.

:hidden:`update_experiment`
---------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.update_experiment
   :noindex:

:hidden:`fetch_experiments`
---------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.fetch_experiments
   :noindex:

:hidden:`delete_experiment`
---------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.delete_experiment
   :noindex:

:hidden:`register_trial`
------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.register_trial
   :noindex:

:hidden:`reserve_trial`
-----------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.reserve_trial
   :noindex:

:hidden:`fetch_trials`
----------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.fetch_trials
   :noindex:

:hidden:`delete_trials`
-----------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.delete_trials
   :noindex:

:hidden:`get_trial`
-------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.get_trial
   :noindex:

:hidden:`update_trials`
-----------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.update_trials
   :noindex:

:hidden:`update_trial`
-----------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.update_trial
   :noindex:

:hidden:`fetch_lost_trials`
---------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.fetch_lost_trials
   :noindex:

:hidden:`fetch_pending_trials`
------------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.fetch_pending_trials
   :noindex:

:hidden:`fetch_noncompleted_trials`
-----------------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.fetch_noncompleted_trials
   :noindex:

:hidden:`fetch_trials_by_status`
--------------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.fetch_trials_by_status
   :noindex:

:hidden:`count_completed_trials`
--------------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.count_completed_trials
   :noindex:

:hidden:`count_broken_trials`
-----------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.count_broken_trials
   :noindex:

:hidden:`set_trial_status`
--------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.set_trial_status
   :noindex:


.. _database_backend:

Database
~~~~~~~~

.. warning::

   The database backends are not meant to be used directly by users.
   Be careful if you use any method which modifies the data in database or
   you may break your experiment or trials.

The database backend used to be the sole database support
initially. An additional abstraction layer, the storage protocol,
has been added with the goal to support various storage types
such as third-party experiment management platforms which
could not be supported using the basic methods ``read``
and ``write``.
This is why the database backend has been turned into
a legacy storage procotol. Because it is the default
storage protocol, we document it here for users
who may need even more flexibility than what the
storage protocol provides.

There is two ways for creating the database client. If you
already created an experiment client, the database
was already created during the process of creating the
experiment client and you can get it with
:py:func:`orion.storage.legacy.get_database`.
Otherwise, you can create the database client with
:py:func:`orion.storage.legacy.setup_database` before
fetching it with
:py:func:`get_database() <orion.storage.legacy.get_database>`.
To recap, you can create it indirectly with
:py:func:`create_experiment() <orion.client.create_experiment>`
or directly with
:py:func:`setup_database() <orion.storage.legacy.setup_database>`.
In both case, you can access it with
:py:func:`get_database() <orion.storage.legacy.get_database>`.

Here's an example on how you could remove an experiment

.. code-block:: python

   from orion.client import create_experiment
   from orion.storage.legacy import get_database, setup_database

   # Create the ExperimentClient and database implicitly
   experiment = create_experiment('exp-name', space=dict(x='uniform(0, 1)'))

   # Or create database explicitly using setup_database
   setup_database(dict(
       type='pickleddb',
       host='db.pkl'
       )
   )

   # This gets the db singleton that was already instantiated within the experiment object.
   db = get_database()

   # To remove all trials of an experiment
   db.remove('trials', dict(experiment=experiment.id))

   # To remove the experiment
   db.remove('experiments', dict(_id=experiment.id))


:hidden:`read`
--------------

.. automethod:: orion.core.io.database.Database.read
   :noindex:

:hidden:`write`
---------------

.. automethod:: orion.core.io.database.Database.write
   :noindex:

:hidden:`remove`
----------------

.. automethod:: orion.core.io.database.Database.remove
   :noindex:

:hidden:`read_and_write`
------------------------

.. automethod:: orion.core.io.database.Database.read_and_write
   :noindex:
