.. _install_database:

**************
Database Setup
**************

Oríon needs a database to operate. It is where we store the optimization execution and results data
to enable fast, seamless, and scalable storage integration.

Out of the box, we support three database backend:

#. :ref:`EphemeralDB Config`, an in-memory database
#. :ref:`PickledDB Config`, a file-based database (default)
#. :ref:`MongoDB Config`, a document-oriented database

In this document, we'll review the different methods by which you can configure which database Oríon
will use during its execution. This page also contains the :ref:`installation instructions for
MongoDB <mongodb_install>` and how to :ref:`upgrade the database <upgrade_database>` if necessary.

.. _Database Configuration:

Configuring the database
========================

There are different ways that database backend attributes can be configured. The first way is by
using a global configuration file, which can easily be done using the command ``orion db setup``.
This will create a yaml file of the following format.

   .. code-block:: yaml

      storage:
        database:
          type: 'pickleddb'
          host: '/path/to/a/file.pkl'

The file is typically located at ``$HOME/.config/orion.core/orion_config.yaml`` but it may differ
based on your operating system.

The second way of configuring the database backend is to use environment variables such as

   .. code-block:: sh

       ORION_DB_ADDRESS=/path/to/a/file.pkl
       ORION_DB_TYPE=PickledDB

Note that both configuration methods can be used together, environment variables that are set will
overwrite the corresponding values in the global configuration. This is useful if you need to define
some of them dynamically, such as picking the database port randomly at runtime based on port
availability for ssh tunnels.

The third configuration method is to use a local configuration file which will be passed to Oríon
using the ``--config`` argument.

   .. code-block:: sh

       orion hunt --config=my_local_config.yaml...

As described above, local configuration file can be used in combination with global and environment
variable definitions. Local configuration values will overwrite configuration from both other
methods.

.. _Test Connection:

Testing the configuration
-------------------------

Once you specified a database, use the command ``orion db test`` to test that the configuration is
correct.

The test goes through 3 phases. First one is the aggregation of the configuration across
global, environment variable and local configuration (note that you can pass ``--config`` to include
a local configuration in the tests). The tests will print the resulting configuration at each
stage.

.. code-block:: sh

   $ orion db test

   Check for a configuration inside the default paths...
       {'name': 'orion', 'type': 'pickleddb', 'host': '', 'port': 27017}
   Success
   Check if configuration file has valid database configuration...
       {'name': 'orion', 'type': 'pickleddb', 'host': '', 'port': 27017}
   Success

   [...]

Alternatively, here's an example including all configuration methods.
This is with MongoDB since there are more options to play with.

.. code-block:: sh

   $ ORION_DB_NAME=test
   $ orion db test --config local.yaml

   Check for a configuration inside the global paths...
       {'name': 'test', 'type': 'pickleddb', 'host': '', 'port': 27017}
   Success
   Check if configuration file has valid database configuration...
       {'type': 'mongodb', 'host': 'localhost'}
   Success

   [...]

The second phase tests the creation of the database, which prints out the final configuration
that will be used and then prints the instance created to confirm the database type.

.. code-block:: sh

   $ orion db test

   [...]

   Using configuration: {'name': 'orion', 'type': 'pickleddb', 'host': '', 'port': 27017}
   Check if database of specified type can be created... Success
   DB instance <orion.core.io.database.pickleddb.PickledDB object at 0x7f86d70067f0>

   [...]

The third phase verifies if all operations are supported by the database. It is possible that these
tests fail because of insufficient user access rights on the database.

.. code-block:: sh

   $ orion db test

   [...]

   Check if database supports write operation... Success
   Check if database supports read operation... Success
   Check if database supports count operation... Success
   Check if database supports delete operation... Success

.. _Supported Databases:

Supported databases
===================

In this section, we show snippets of configuration for each database backend.

.. _EphemeralDB Config:

EphemeralDB
-----------

:ref:`EphemeralDB <EphemeralDB>` is the `in-memory` database used when executing Oríon with the
argument ``--debug``. It is wiped out of memory at end of the execution.

.. code-block:: yaml

   database:
      type: 'ephemeraldb'

Arguments
~~~~~~~~~

EphemeralDB has no arguments.

.. _PickledDB Config:

PickledDB
---------

PickledDB_ is recommended for its simplicity to setup but it is generally not suited
for parallel optimization with more than 50 workers. This is however just a rule of thumb and
you may find PickledDB to work properly with more workers if your tasks take a significant
amount of time to execute.

.. code-block:: yaml

   database:
      type: 'pickleddb'
      host: '/path/to/a/save/file.pkl'

.. _PickledDB: https://pythonhosted.org/pickleDB/

Arguments
~~~~~~~~~

.. list-table::

   * - ``host``
     - File path where the database is saved. All workers require access to this file for parallel
       optimization so make sure it is on a shared file system.

.. _MongoDB Config:

MongoDB
-------

MongoDB_ is the recommended backend for large-scale parallel optimizations, where the number of
workers gets higher than 50. Make sure to review our :ref:`MongoDB installation instructions
<mongodb_install>`.

.. code-block:: yaml

   database:
      type: 'mongodb'
      name: 'orion_test'
      host: 'mongodb://user:pass@localhost'

.. _MongoDB: https://www.mongodb.com/

Arguments
~~~~~~~~~

.. list-table::

   * - ``name``
     - Name of the mongodb database.
   * - ``host``
     - Can be either the host address  (hostname or IP address) or a mongodb URI. Default is ``localhost``.
   * - ``port``
     - Port that database servers listens to for requests. Default is 27017.

.. _mongodb_install:

Installing MongoDB
==================

To install MongoDB locally, follow the `official instructions
<https://docs.mongodb.com/manual/administration/install-community/>`_ for your operating system.
Alternatively, use :ref:`MongoDB Atlas <mongodb-atlas>` to create a database in the cloud.

Once MongoDB is installed, create the database using:

.. code-block:: sh

   $ mongo orion_test --eval 'db.createUser({user:"user",pwd:"pass",roles:["readWrite"]});'

.. _mongodb-atlas:

MongoDB Atlas
-------------

MongoDB Atlas is a cloud-hosted MongoDB service on AWS, Azure and Google Cloud. Deploy, operate, and
scale a MongoDB database in just a few clicks.

1. Create an account `here <https://www.mongodb.com/cloud/atlas>`_.
2. Follow the defaults to create a free cluster.
3. Add cluster name and click on "Create Cluster".
4. Wait for the cluster to be created.
5. In "Overview" tab, click on "CONNECT".
6. Add the IP of your computer to the whitelist or "Allow access from anywhere."
7. Click on "Connect your application".
8. Orion supports MongoDB drive 3.4, so choose driver 3.4.
9. Copy the generated SRV address and replace "USERNAME" and "PASSWORD" with your
   Atlas MongoDB username and password.
10. To test, move to the first page, select "connect", and then choose "Connect
    with your the Mongo Shell". Select your operating system and copy the URL:

    .. code-block:: sh

      mongo YOUR_URL --username YOUR_USER_NAME

11. Configure Oríon's YAML file (See :ref:`Database Configuration`).

.. _upgrade_database:

Upgrading the database
======================

The database's schema may change between major version of Oríon. If this happens, you will get the
following error after upgrading Oríon.

.. code-block:: sh

   The database is outdated. You can upgrade it with the command `orion db upgrade`.

**Before upgrading the database**, make sure to create a backup of it. You should also make sure
that there is no process writing to the database during the upgrade otherwise the latter could fail
and corrupt the database.

When ready, simply run the upgrade command ``orion db upgrade``.
