**************
Running on HPC
**************

This guide is based on the example described in :doc:`/tutorials/pytorch-mnist`.

Parallel optimization using arrays
==================================

For simplicity, we will only use Slurm in the examples, but the same applies for PBS based systems
with the argument ``-t``.

Oríon synchronises workers transparently based on the experiment name. Thanks to this there is no
master to setup and we can focus solely on submitting the workers. Also, since all
synchronisation is done through the database, there is no special setup required to connect workers
together. A minimal Slurm script to launch 10 workers would thus only require the following 2 lines.

.. code-block:: bash

    #SBATCH --array=1-10

    orion hunt -n parallel-exp python main.py --lr~'loguniform(1e-5, 1.0)'

All workers are optimizing the experiment ``parallel-exp`` in parallel, each holding a copy of the
optimization algorithm. Adding Slurm options to execute the mnist example with proper resources
gives the following

.. code-block:: bash

    #SBATCH --array=1-10
    #SBATCH --cpus-per-task=2
    #SBATCH --output=/path/to/some/log/parallel-exp.%A.%a.out
    #SBATCH --error=/path/to/some/log/parallel-exp.%A.%a.err
    #SBATCH --gres=gpu:1
    #SBATCH --job-name=parallel-exp
    #SBATCH --mem=10GB
    #SBATCH --time=2:59:00

    orion hunt -n parallel-exp --worker-trials 1 python main.py --lr~'loguniform(1e-5, 1.0)'

For now, Oríon does not provide detection of lost trials if a worker gets killed due to a
timeout. Such trial would be indefinitely marked as ``pending`` in the DB and thus could not be
executed again unless the state is fixed manually. To avoid this, you can set the timeout large
enough for a single trial and use the argument ``--worker-trials 1`` to limit worker to
execute only one trial and then quit. If you have a large amount of tasks to execute but do not want
to have as many workers, you can limit the number of simultaneous jobs with the
character ``%`` (ex: ``#SBATCH --array=1-100%10``).

.. code-block:: bash

    #SBATCH --array=1-100%10
    #SBATCH --cpus-per-task=2
    #SBATCH --output=/path/to/some/log/parallel-exp.%A.%a.out
    #SBATCH --error=/path/to/some/log/parallel-exp.%A.%a.err
    #SBATCH --gres=gpu:1
    #SBATCH --job-name=parallel-exp
    #SBATCH --mem=10GB
    #SBATCH --time=2:59:00

    orion hunt -n parallel-exp --worker-trials 1 python main.py --lr~'loguniform(1e-5, 1.0)'


SSH tunnels
===========

.. note:

   MongoDB does not play nicely with ssh tunnels. You can try using ``PickledDB`` instead, following
   the configuration steps describded :ref:`here <Database Configuration>`.

Some HPC infrastructure does not provide access to internet from the compute nodes. To get access to
the database from the compute nodes, it is necessary to open ssh tunnels to a gateway (typically
login nodes). The ssh tunnel will redirect traffic from different address and port, therefore the
config of the database needs to be modified accordingly. Suppose our config was the following
without using an ssh tunnel. (``$HOME/.config/orion.core/orion_config.yaml``)

.. code-block:: yaml

      database:
        type: 'mongodb'
        name: 'db_name'
        host: 'mongodb://user:pass@<db address>:27017'

Using port 42883, the config would now be like this

.. code-block:: yaml

      database:
        type: 'mongodb'
        name: 'db_name'
        host: 'mongodb://user:pass@localhost'
        port: '42883'

Note that the port number was removed from ``host`` because it would have precedence over ``port``.
Also, the host address is changed to ``localhost``, because the traffic is send to
``localhost:42883`` and then transferred to ``<db address>:27017`` on the other end of the ssh
tunnel.

Now, to open the ssh tunnel from the compute node, use this command

.. code-block:: bash

    ssh -o StrictHostKeyChecking=no <gateway address> -L 42883:<db address>:27017 -n -N -f

Where <gateway address> is the hostname of the gateway (login node) that you want to connect to.

This would work for a single job, but it is likely to cause trouble if many jobs end up on the same
compute node. The first job would open the ssh tunnel, and the following ones would fail because the
port would no longer be available. They would still all be able to use the ssh tunnel, however when
the first job would end, the ssh tunnel would close with it and all following jobs would loose
access to the DB. To get around this problem, we need to randomly choose available ports instead,
so that two jobs working on the same node use different ports. Here's how

.. code-block:: bash


    export ORION_DB_PORT=$(python -c "from socket import socket; s = socket(); s.bind((\"\", 0)); print(s.getsockname()[1])")

    ssh -o StrictHostKeyChecking=no <gateway address> -L $ORION_DB_PORT:<db address>:27017 -n -N -f

These lines can then be added to the script to submit workers in parallel.

.. code-block:: bash

    #SBATCH --array=1-100%10
    #SBATCH --cpus-per-task=2
    #SBATCH --output=/path/to/some/log/parallel-exp.%A.%a.out
    #SBATCH --error=/path/to/some/log/parallel-exp.%A.%a.err
    #SBATCH --gres=gpu:1
    #SBATCH --job-name=parallel-exp
    #SBATCH --mem=10GB
    #SBATCH --time=2:59:00

    export ORION_DB_PORT=$(python -c "from socket import socket; s = socket(); s.bind((\"\", 0)); print(s.getsockname()[1])")

    ssh -o StrictHostKeyChecking=no <gateway address> -L $ORION_DB_PORT:<db address>:27017 -n -N -f

    orion hunt -n parallel-exp --worker-trials 1 python main.py --lr~'loguniform(1e-5, 1.0)'


Notes for MongoDB
-----------------

You may experience problems with MongoDB if you are using an encrypted connection with SSL and
if you are using replica sets
(both of which are highly recommended for security and high availability).

SSL
~~~

You will need to set the variable
``ssl_match_hostname=false`` in your URI to bypass the SSL hostname check. This is because
the address used with the tunnel is ``localhost`` and this won't be recognised by your SSL
certificate. From pymongo's documentation

    Think very carefully before setting this to False as that could make your application
    vulnerable to man-in-the-middle attacks

Replica Sets
~~~~~~~~~~~~

So far, we know no simple methods to use replica sets with ssh tunnels and therefore we cannot
recommend anything better than not setting up replica set in your MongoDB servers if you need to use
ssh tunnels.  When dealing with replica sets, the local process tries to open direct connection to
each secondary servers (replica sets), which are normally on different hosts. These connections,
which are pointing to different addresses, cannot pass through the ssh tunnel that was opened for
the address of the primany mongodb server.
