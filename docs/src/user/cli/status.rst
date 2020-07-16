``status`` Overview of trials for experiments
---------------------------------------------

When you reach a certain amount of trials, it becomes hard to keep track of them. This is where the
``status`` command comes into play. The ``status`` command outputs the status of the different
trials inside every experiment or a specific EVC tree. It can either give you an overview of
the different trials status, i.e., the number of currently ``completed`` trials and so on, or, it
can give you a deeper view of the experiments by outlining every single trial, its status and its
objective.

Basic Usage
~~~~~~~~~~~
The most basic of usages is to simply run ``status`` without any other arguments, except for a local
configuration file if needed. This will then output a list of all the experiments inside your
database with the count of every type of trials related to them. If an experiment has at least one
``completed`` trial associated with it, the objective value of the best one will be printed as well.
Children experiments are printed below their parent and are indicated through a different tab
alignment than their parent, mainly, one tab further. This continues on for grand-children, and so
on and so forth. We provide an example output to illustrate this:

.. code-block:: bash

    root-v1
    =======
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed           5                  4534.95


      child_1-v1
      ==========
      status       quantity    min example_objective
      ---------  ----------  -----------------------
      completed           5                  4547.28


    other_root-v1
    =============
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed           5                  4543.73

The ``--all`` Argument
~~~~~~~~~~~~~~~~~~~~~~
The basic status command combines statistics of all trials for each status. However, if you want to
see every individual trial, with its id and its status, you can use the ``--all``
argument which will print out every single trial for each experiment with their full information.
Here is a sample output using the same experiments and trials as before:

.. code-block:: console

    orion status --all

.. code-block:: bash

    root-v1
    =======
    id                                status       min example_objective
    --------------------------------  ---------  -----------------------
    bc222aa1705b3fe3a266fd601598ac41  completed                  4555.13
    59bf4c85305b7ba065fa770805e93cb1  completed                  4653.49
    da5159d9d36ef44879e72cbe7955347c  completed                  4788.08
    2c4c6409d3beeb179fce3c83b4f6d8f8  completed                  4752.45
    6a82a8a55d2241d989978cf3a7ebbba0  completed                  4534.95


      child_1-v1
      ==========
      id                                status       min example_objective
      --------------------------------  ---------  -----------------------
      a3395b7192eee3ca586e93ccf4f12f59  completed                  4600.98
      e2e2e96e8e9febc33efb17b9de0920d1  completed                  4786.43
      7e0b4271f2972f539cf839fbd1b5430d  completed                  4602.58
      568acbcb2fa3e00c8607bdc2d2bda5e3  completed                  4748.09
      5f9743e88a29d0ee87b5c71246dbd2fb  completed                  4547.28


    other_root-v1
    =============
    id                                status       min example_objective
    --------------------------------  ---------  -----------------------
    aaa16658770abd3516a027918eb91be5  completed                  4761.33
    68233ce61ee5edfb6fb029ab7daf2db7  completed                  4543.73
    cc0b0532c56c56fde63ad06fd73df63f  completed                  4753.5
    b5335589cb897bbea2b58c5d4bd9c0c1  completed                  4751.15
    a4a711389844567ac1b429eff96964e4  completed                  4790.87



The ``--collapse`` Argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~
On the other hand, if you wish to only get an overview of the experiments and the amount of trials
linked to them without looking through the whole EVC tree, you can use the ``--collapse``
option. As its name indicates, it will collapse every children into the root experiment and make a
total count of the amount of trials `in that EVC tree`. As always, we provide an output to
give you an example:


.. code-block:: console

    orion status --collapse

    root-v1
    =======
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed          10                  4534.95


    other_root-v1
    =============
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed           5                  4543.73


The ``--name`` Argument
~~~~~~~~~~~~~~~~~~~~~~~
If you wish to isolate a single EVC tree and look at their trials instead of listing every
single experiments, you can use the ``--name`` argument by itself or combine it with the ones above
to obtain the same results, but constrained. Once again, some examples for each type of scenrario is
given:

.. code-block:: console

    orion status --name root

.. code-block:: bash

    root-v1
    =======
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed          10                  4534.95


      child_1-v1
      ==========
      status       quantity    min example_objective
      ---------  ----------  -----------------------
      completed          10                  4547.28

.. code-block:: console

    orion status --name root --all

.. code-block:: bash

    root-v1
    =======
    id                                status       min example_objective
    --------------------------------  ---------  -----------------------
    bc222aa1705b3fe3a266fd601598ac41  completed                  4555.13
    59bf4c85305b7ba065fa770805e93cb1  completed                  4653.49
    da5159d9d36ef44879e72cbe7955347c  completed                  4788.08
    2c4c6409d3beeb179fce3c83b4f6d8f8  completed                  4752.45
    6a82a8a55d2241d989978cf3a7ebbba0  completed                  4534.95


      child_1-v1
      ==========
      id                                status       min example_objective
      --------------------------------  ---------  -----------------------
      a3395b7192eee3ca586e93ccf4f12f59  completed                  4600.98
      e2e2e96e8e9febc33efb17b9de0920d1  completed                  4786.43
      7e0b4271f2972f539cf839fbd1b5430d  completed                  4602.58
      568acbcb2fa3e00c8607bdc2d2bda5e3  completed                  4748.09
      5f9743e88a29d0ee87b5c71246dbd2fb  completed                  4547.28

.. code-block:: console

    orion status --name root --collapse

.. code-block:: bash

    root-v1
    =======
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed          10                  4534.95


``status`` and the experiment tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `status` command handles the experiment tree in a particular fashion. Since most users will
simply use the incrementing version mechanism instead of constantly renaming their experiments, the
experiment tree can grow large in depth but not in breadth. This leads to a very hard to read output
if the command was to print such a tree in the same way as presented above. Instead, if a root
experiment does not have any children named differently, i.e. its tree only contains different
version of itself, `status` will only print the latest version. However, if any of its children
is named differently, than the whole tree will be printed just like above.

To illustrate the first case, suppose we have an experiment named `test` with three different
versions: version `1`, `2` and `3`. Then running status as usual will only output version `3`.

.. code-block:: console

    orion status --name test

.. code-block:: bash

    test-v3
    =======
    empty

The ``--version`` argument
^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``--version`` argument allows you to specify a version to print instead of getting the latest
one.  Suppose we have the same setup as above with three experiments named ``test`` but with
different versions. Then running the following command will output the second version instead of the
latest.

.. code-block:: console

    orion status --name test --version 2

.. code-block:: bash

    test-v2
    =======
    empty

It should be noted that using ``--version`` with any of ``--collapse`` or ``--expand-versions``
will lead to a ``RuntimeError``.

The ``--expand-versions`` argument
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As specified above, if there are no children of a root experiment with a different name then the
experiment tree will not be printed in its entirety. The ``--expand-versions`` allows you to get the
full output of the experiment tree, regardless if it only contains different versions. Once again,
suppose we have the same setup with experiment `test`, then running the following command will print
the experiment tree.

.. code-block:: console

    orion status --name test --expand-versions

.. code-block:: bash

    test-v1
    =======
    empty


      test-v2
      =======
      empty


        test-v3
        =======
        empty
