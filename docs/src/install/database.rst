.. _install_database:

**************
Setup Database
**************

.. note::

   You can avoid the complexity of setting up a MongoDB server and give a try to the simple
   alternative we are currently integrating in Oríon following the configuration steps
   :ref:`here <Database Configuration>` for :ref:`PickledDB <PickledDB Config>`.
   We plan to make PickledDB the default database backend in release v0.2.0.

We are currently using a MongoDB_ dependent API
to persistently record history changes and have it serve as
a central worker to the asynchronous communication between the
workers who produce and evaluate suggested points in a problem's
parameter space.

Hence, an installation requirement to MongoDB_ and the setup of a database, if
not intending to use an already existing one, is necessary.

We are going to follow an example of installing and setting up a minimal
database locally.

Local MongoDB Installation
==========================

Supposing we are in a Linux machine, follow the installation process
(preferably respecting the package manager of your distribution) discussed in
`Mongo docs <https://docs.mongodb.com/manual/administration/install-on-linux/>`__. If
your Linux distribution is not enlisted in this link, then follow the preferred
way described in your distribution's web pages.

.. note::
   Good or useful starting references can be found in:

   * `Mongo Shell Quick Reference <https://docs.mongodb.com/manual/reference/mongo-shell/>`_
   * `Tutorialspoint <https://www.tutorialspoint.com/mongodb/mongodb_create_database.htm>`_
   * `ArchLinux wiki <https://wiki.archlinux.org/index.php/MongoDB>`_

Setup MongoDB without root access
---------------------------------

As mentioned in  `Mongo docs <https://docs.mongodb.com/manual/tutorial/install-mongodb-on-debian/#using-tgz-tarballs>`__ download MongoDB, extract it and make sure the binaries are in a directory listed in your PATH environment variable. Next create the database using::

      mongo orion_test --eval 'db.createUser({user:"user",pwd:"pass",roles:["readWrite"]});'

To start MongoDb, create a directory to contain the database::

      mongod --dbpath /path/to/database

Setup MongoDB with root access
------------------------------
Follow the instructions described in  `Mongo docs <https://docs.mongodb.com/manual/administration/install-on-linux/>`_. If you have root access you can invoke the following command as you can read `here <https://docs.mongodb.com/manual/reference/method/db.createUser/>`__::

   mongo orion_test --eval 'db.createUser({user:"user",pwd:"pass",roles:["readWrite"]});'

And start MongoDB::

   sudo service mongod start

Atlas MongoDB
=============
1. Create an account `here <https://www.mongodb.com/cloud/atlas>`_.
2. Follow the defaults to create a free cluster.
3. Add cluster name and click on "Create Cluster".
4. Wait for the cluster to be created.
5. In "Overview" tab, click on "CONNECT".
6. Add the IP of your compuer to the whitelist or "Allow access from anywhere."
7. Click on "Connect your application".
8. Orion supports MongoDB drive 3.4, so choose driver 3.4.
9. Copy the generated SRV address and replace "USERNAME" and "PASSWORD" with your
   Atlas MongoDB username and password.
10. To test, move to the first page, select "connect", and then choose "Connect
    with your the Mongo Shell". Select your operating system and copy the URL:

    .. code-block:: sh

      mongo YOUR_URL --username YOUR_USER_NAME

11. Configure Oríon's YAML file (See next section).


.. _Database Configuration:

Configuring Oríon's Database
============================

There are different ways that database backend attributes can be configured.
The first one is by using a global configuration file, which can easily be done
using the command ``orion db setup``. This will create a yaml file
of the following format.

   .. code-block:: yaml

      database:
        type: 'mongodb'
        name: 'orion_test'
        host: 'mongodb://user:pass@localhost'

The file is typically located at ``$HOME/.config/orion.core/orion_config.yaml`` but it may differ
based on your operating system.

The second way of configuring the database backend is to use environment variables such as

   .. code-block:: sh

       ORION_DB_ADDRESS=mongodb://user:pass@localhost
       ORION_DB_NAME=orion_test
       ORION_DB_TYPE=MongoDB
       ORION_DB_PORT=27017

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

MongoDB
-------

   .. code-block:: yaml

      database:
        type: 'mongodb'
        name: 'orion_test'
        host: 'mongodb://user:pass@localhost'

MongoDB backend is the recommended one for large scale parallel optimisation, where
number of workers gets higher than 50.

Arguments
~~~~~~~~~

``name``

Name of the mongodb database.

``host``

Can be either the host address  (hostname or IP address) or a mongodb URI. Default is ``localhost``.

``port``

Port that database servers listens to for requests. Default is 27017.



.. _PickledDB Config:

PickledDB
---------

   .. code-block:: yaml

      database:
        type: 'pickleddb'
        host: '/some/path/to/a/file/to/save.pkl'

PickledDB is recommended for its simplicity to setup but it is generally not suited
for parallel optimisation with more than 50 workers. This is however just a rule of thumb and
you may find PickledDB to work properly with more workers if your tasks take a significant
amount of time to execute.

Arguments
~~~~~~~~~

``host``

File path where the database is saved. All workers require access to this file for parallel
optimisation so make sure it is on a shared file system.

EphemeralDB
-----------

   .. code-block:: yaml

      database:
        type: 'ephemeraldb'

EphemeralDB is the `in-memory` database used when executing Oríon with the argument
``--debug``. It is wiped out of memory at end of execution.

EphemeralDB has no arguments.

Test connection
===============

You can use the command ``orion db test`` to test the setup of your database backend.

.. code-block:: sh

   $ orion db test

   Check for a configuration inside the default paths...
       {'type': 'mongodb', 'name': 'mydb', 'host': 'localhost'}
   Success
   Check for a configuration inside the environment variables... Skipping
   No environment variables found.
   Check if configuration file has valid database configuration... Skipping
   Missing configuration file.
   Using configuration: {'type': 'mongodb', 'name': 'mydb', 'host': 'localhost'}
   Check if database of specified type can be created... Success
   DB instance <orion.core.io.database.mongodb.MongoDB object at 0x7f86d70067f0>
   Check if database supports write operation... Success
   Check if database supports read operation... Success
   Check if database supports count operation... Success
   Check if database supports delete operation... Success

The tests goes throught 3 phases. First one is the aggregation of the configuration across
global, environment variable and local configuration (note that you can pass ``--config`` to include
a local configuration in the tests). The tests will print the resulting configuration at each
stage. Here's an example including all three configuration methods.

.. code-block:: sh

   $ ORION_DB_PORT=27018 orion db test --config local.yaml

   Check for a configuration inside the global paths...
       {'type': 'mongodb', 'name': 'mydb', 'host': 'localhost'}
   Success
   Check for a configuration inside the environment variables...
       {'type': 'mongodb', 'name': 'mydb', 'host': 'localhost', 'port': '27018'}
   Success
   Check if configuration file has valid database configuration...
       {'type': 'mongodb', 'name': 'mydb', 'host': 'localhost', 'port': '27017'}
   Success

The second phase is the creation of the database, which prints out the final configuration
that will be used and then prints the instance created to confirm the database type.

.. code-block:: sh

   $ orion db test

   [...]

   Using configuration: {'type': 'mongodb', 'name': 'mydb', 'host': 'localhost'}
   Check if database of specified type can be created... Success
   DB instance <orion.core.io.database.mongodb.MongoDB object at 0x7f86d70067f0>

The third phase verifies if all operations are supported by the database. It is possible that these
tests fail because of insufficient user access rights on the database.

.. code-block:: sh

   $ orion db test

   [...]

   Check if database supports write operation... Success
   Check if database supports read operation... Success
   Check if database supports count operation... Success
   Check if database supports delete operation... Success


Upgrade Database
================

Database scheme may change from one version of Oríon to another. If such change happens, you will
get the following error after upgrading Oríon.

.. code-block:: sh

   The database is outdated. You can upgrade it with the command `orion db upgrade`.

Make sure to create a backup of your database before upgrading it. You should also make sure that no
process writes to the database during the upgrade otherwise the latter could fail. When ready,
simply run the upgrade command.

.. code-block:: sh

   orion db upgrade
