.. contents:: User's Guide 104: Database

**************
Setup Database
**************

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

Local Installation
==================

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


Configuring Oríon's Database
============================

There are two possible ways that database attributes can be configured.
The first one is by using environmental variables and the second one is by using
Oríon configuration files.

   1. By setting appropriate environmental variables of the shell used to call
      Oríon's executable.

   .. code-block:: sh

      export ORION_DB_ADDRESS=mongodb://user:pass@localhost
      export ORION_DB_NAME=orion_test
      export ORION_DB_TYPE=MongoDB

   2. By creating a section in an Oríon's configuration YAML file, like `this one <https://github.com/epistimio/orion/blob/master/tests/functional/demo/orion_config_random.yaml>`_
      used by our functional tests.

   .. code-block:: yaml

      database:
        type: 'mongodb'
          name: 'orion_test'
          host: 'mongodb://user:pass@localhost'

As it will be referenced with detail in :doc:`configuration's documentation </user/configuring>`,
the environmental variable definitions precede the ones within files in default
locations, and configuration files provided via executable's cli precede
environmentals.

.. _MongoDB: https://www.mongodb.com/
