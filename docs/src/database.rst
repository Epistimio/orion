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
`Mongo docs <https://docs.mongodb.com/manual/administration/install-on-linux/>`_. If
your Linux distribution is not enlisted in this link, then follow the preferred
way described in your distribution's web pages.

.. note::
   Good or useful starting references can be found in:

   * `Mongo Shell Quick Reference <https://docs.mongodb.com/manual/reference/mongo-shell/>`_
   * `Tutorialspoint <https://www.tutorialspoint.com/mongodb/mongodb_create_database.htm>`_
   * `ArchLinux wiki <https://wiki.archlinux.org/index.php/MongoDB>`_

Create a MongoDB
----------------

Invoke the following command as you can read `here <https://docs.mongodb.com/manual/reference/method/db.createUser/>`_::

   mongo orion_test --eval 'db.createUser({user:"user",pwd:"pass",roles:["readWrite"]});'

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

   2. By creating a section in an Oríon's configuration YAML file, like `this one <https://github.com/mila-udem/orion/blob/master/tests/functional/demo/orion_config_random.yaml>`_
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



Atlas MongoDB
=============

https://www.mongodb.com/cloud/atlas


NEXT: CLUSTER TIER

NEXT: ADDITIONAL SETTINGS

NEXT: CLUSTER NAME

Some cluster name...

Create Cluster

Security tab
MongoDB users

user name
autogen pass: Hx6yN2Dp40GgTirF


https://docs.atlas.mongodb.com/getting-started/

IP whitelist
(option: Add current ip address)


On Clusters/Overview tab:
Connect



https://docs.atlas.mongodb.com/mongo-shell-connection/

Test with
mongo "mongodb+srv://orion-efjp0.mongodb.net/test" --username bouthilx


Connect Your Application
Orion supports MongoDB drive 3.4
Copy the address

database:
    type: 'mongodb'
    name: 'name_of_db'
    host: <URI>

Done.
