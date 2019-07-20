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

    root
    ====
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed           5                  4534.95


      child_1
      =======
      status       quantity    min example_objective
      ---------  ----------  -----------------------
      completed           5                  4547.28


    other_root
    ==========
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

    root
    ====
    id                                status       min example_objective
    --------------------------------  ---------  -----------------------
    bc222aa1705b3fe3a266fd601598ac41  completed                  4555.13
    59bf4c85305b7ba065fa770805e93cb1  completed                  4653.49
    da5159d9d36ef44879e72cbe7955347c  completed                  4788.08
    2c4c6409d3beeb179fce3c83b4f6d8f8  completed                  4752.45
    6a82a8a55d2241d989978cf3a7ebbba0  completed                  4534.95


      child_1
      =======
      id                                status       min example_objective
      --------------------------------  ---------  -----------------------
      a3395b7192eee3ca586e93ccf4f12f59  completed                  4600.98
      e2e2e96e8e9febc33efb17b9de0920d1  completed                  4786.43
      7e0b4271f2972f539cf839fbd1b5430d  completed                  4602.58
      568acbcb2fa3e00c8607bdc2d2bda5e3  completed                  4748.09
      5f9743e88a29d0ee87b5c71246dbd2fb  completed                  4547.28


    other_root
    ==========
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

    root
    ====
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed          10                  4534.95


    other_root
    ==========
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

    root
    ====
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed          10                  4534.95


      child_1
      =======
      status       quantity    min example_objective
      ---------  ----------  -----------------------
      completed          10                  4547.28

.. code-block:: console

    orion status --name root --all

.. code-block:: bash

    root
    ====
    id                                status       min example_objective
    --------------------------------  ---------  -----------------------
    bc222aa1705b3fe3a266fd601598ac41  completed                  4555.13
    59bf4c85305b7ba065fa770805e93cb1  completed                  4653.49
    da5159d9d36ef44879e72cbe7955347c  completed                  4788.08
    2c4c6409d3beeb179fce3c83b4f6d8f8  completed                  4752.45
    6a82a8a55d2241d989978cf3a7ebbba0  completed                  4534.95


      child_1
      =======
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

    root
    ====
    status       quantity    min example_objective
    ---------  ----------  -----------------------
    completed          10                  4534.95
