.. role:: hidden
    :class: hidden-section


.. _storage:

*******
Storage
*******

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
================

The experiment client must be created with the helper function
:py:func:`create_experiment() <orion.client.create_experiment>` which will take care of
initiating the storage backend and create a new experiment if non-existant or simply load
the corresponding experiment from the storage.

There is a small subset of methods to fetch trials or create new ones. We focus here
on the methods for loading or creation of trials in particular, see
:py:class:`ExperimentClient <orion.client.experiment.ExperimentClient>` for documentation
of all methods.

Here is a short example to fetch trials or insert a new one.

.. code-block:: python

   from orion.client import create_experiment

   # Create the ExperimentClient
   experiment = create_experiment('exp-name', space=dict(x='uniform(0, 1)'))

   # To fetch all trials from an experiment
   trials = experiment.fetch_trials()

   # Insert a new trial in storage
   experiment.insert(dict(x=0.5))

   # Insert a new trial and reserve to execute
   trial = experiment.insert(dict(x=0.6), reserve=True)

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
=======

.. warning::

   The storage backends are not meant to be used directly by users.
   Be careful if you use any method which modifies the data in storage or
   you may break your experiment or trials.

The storage backend is used by the
:py:class:`ExperimentClient <orion.client.experiment.ExperimentClient>`
to read and write persistant records of the experiment and trials.
Although we recommand using the experiment client,
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

:hidden:`fetch_experiment`
--------------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.fetch_experiments
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

:hidden:`get_trial`
-------------------

.. automethod:: orion.storage.base.BaseStorageProtocol.get_trial
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
========

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

:hidden:`write`
---------------

.. automethod:: orion.core.io.database.Database.write

:hidden:`remove`
----------------

.. automethod:: orion.core.io.database.Database.remove

:hidden:`read_and_write`
------------------------

.. automethod:: orion.core.io.database.Database.read_and_write
