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

.. note::

   This is the same database required to be setup in order to run the tests.

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
using the command ``orion setup``. This will create a yaml file
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

File path where the database is saved. All workers requires access to this file for parallel
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
---------------

You can first check that everything works as expected by testing with the
``debug`` mode. This mode bypass the database in the configuration. If you run
the following command, you should get the following error.

.. code-block:: bash

    $ orion --debug hunt -n dummy
    ...
    AttributeError: 'str' object has no attribute 'configuration'

That's a terrible error message. -_- Note to ourselves; Improve this error message. What this should
tell is that the connection to database was successful but Oríon could not find any script to
optimize.

Now remove the option ``--debug`` to test the database. If it fails to connect,
you will get the following error. Otherwise, you'll get the (terrible) error above again
if it succeeded. Note that a connection failure will hang for approximately 60
seconds before giving up.

.. code-block:: bash

    $ orion hunt -n dummy
    ...
    orion.core.io.database.DatabaseError: Connection Failure: database not found on specified uri

If it fails, try running with ``-vv`` and make sure your configuration file is
properly found. Suppose your file path is ``/u/user/.config/orion.config/orion_config.yaml``,
then you should **NOT** see the following line in the output otherwise it means it is not found.

.. code-block:: bash

    DEBUG:orion.core.io.resolve_config:[Errno 2] No such file or directory: '/u/user/.config/orion.config/orion_config.yaml'

When you are sure the configuration file is found, look for the configuration
used by Oríon to initiate the DB connection.

.. code-block:: bash

    DEBUG:orion.core.io.experiment_builder:Creating mongodb database client with args: {'name': 'user', 'host': 'mongodb://user:pass@localhost'}

Make sure you have the proper database name, database type and host URI.


.. _MongoDB: https://www.mongodb.com/
