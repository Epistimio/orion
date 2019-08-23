``list`` Overview of all experiments
------------------------------------

Once you have launched a certain amount of experiments, you might start to lose track of some of
them. You might forget their name and whether or not they are the children of some other experiment
you have also forgotten. In any cases, the ``list`` command for Oríon will help you visualize the
experiments inside your database by printing them in a easy-to-understand tree-like structure.

Configuration
~~~~~~~~~~~~~
As per usual with Oríon, if no configuration file is provided to the ``list`` command, the default
configuration will be used. You can however provide a particular configuration file through the
usual ``-c`` or ``--config`` argument. This configuration file needs only to contain a valid
database configuration.

Basic Usage
~~~~~~~~~~~
The most basic usage of ``list`` is to use it without any arguments. This will simply print out
every experiments inside the database in a tree-like fashion, so that the children of the
experiments are easily identifiable. Here is a sample output to serve as an example:

.. code-block:: console

            ┌child_2-v1
     root-v1┤
            └child_1-v1┐
                       └grand_child_1-v1
     other_root-v1

Here, you can see we have five experiments. Two of them are roots, which mean they do not have any
parents. One of them, ``other_root``, does not have any children and so, it does not have any
branches coming out of it. On the other hand, the ``root`` experiment has multiple children,
``child_1`` and ``child_2``, which are printed on the same tree level, and one grand-child,
``grand_child_1`` which branches from ``child_1``.

The ``--name`` argument
~~~~~~~~~~~~~~~~~~~~~~~
The last example showed you how to print every experiments inside the database in a tree. However,
if you wish to have an overview of the tree of a single experiment, you can add the ``--name``
argument to the call to ``list`` and only the experiment with the provided name and its children
will be shown. Here's two examples using the same set of experiments as above:

.. code-block:: bash

    orion list --name root

Output

.. code-block:: console

            ┌child_2-v1
     root-v1┤
            └child_1-v1┐
                       └grand_child_1-v1

Here, the ``other_root`` experiment is not showned because it is not inside the ``root`` experiment
tree.

.. code-block:: bash

    orion list --name child_1

Output

.. code-block:: console

     child_1-v1┐
               └grand_child_1-v1

Here, the ``root`` and ``child_2`` experiments are not present because they are not children of
``child_1``.
