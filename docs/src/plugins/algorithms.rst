*********************
Developing algorithms
*********************

The documentation here explains how to create new algorithm plugins based on the
`cookiecutter <https://github.com/Epistimio/cookiecutter-orion.algo>`_.

Usage
=====

.. _GitHub: https://github.com/Epistimio/cookiecutter-orion.algo


Install Python requirements to use the template:

.. code-block:: console

    $ python -m pip install cookiecutter>=1.5 versioneer>=0.18 jinja2


Create a new project directly from the template on `GitHub`_:

.. code-block:: console

    $ cookiecutter gh:Epistimio/cookiecutter-orion.algo
    plugin_name []: skopt
    author_name []: Xavier Bouthillier
    author_short [Author Name]:
    author_email []: xavier.bouthillier@umontreal.ca
    github_username []: bouthilx
    copyright [2019, Author Name]:
    short_description [TODO]:
    synopsis [TODO]:
    algo_name []: BayesianOptimizer
    algo_module_name [bayesianoptimizer]: bayes

+-----------------------+--------------------------------------------+--+
| Field                 | Description                                |  |
+-----------------------+--------------------------------------------+--+
| ``plugin_name``       | Will be used for orion.algo.plugin_name    |  |
+-----------------------+--------------------------------------------+--+
| ``author_name``       | For metadata of python package             |  |
+-----------------------+--------------------------------------------+--+
| ``author_short``      | For metadata of python package             |  |
+-----------------------+--------------------------------------------+--+
| ``author_email``      | For metadata of python package             |  |
+-----------------------+--------------------------------------------+--+
| ``github_username``   | Username to build the url for installation |  |
+-----------------------+--------------------------------------------+--+
| ``copyright``         | For the BSD-3 license                      |  |
|                       | (You can change the license)               |  |
+-----------------------+--------------------------------------------+--+
| ``short_description`` | For metadata of python package             |  |
+-----------------------+--------------------------------------------+--+
| ``synopsis``          | For documentation in algo module           |  |
+-----------------------+--------------------------------------------+--+
| ``algo_name``         | Name for the algorithm class               |  |
+-----------------------+--------------------------------------------+--+
| ``algo_module_name``  | Name of the algorithm module               |  |
+-----------------------+--------------------------------------------+--+

This will create the following package structure.

.. code-block:: bash

    orion.algo.{plugin_name}
    ├── README.rst
    ├── setup.cfg
    ├── setup.py
    ├── MANIFEST.in
    ├── LICENSE (BSD License)
    ├── versioneer.py
    ├── tox.ini
    ├── dev-requirements.txt
    ├── doc
    │   ├── requirements.txt
    │   └── src
    │       ├── conf.py
    │       └── index.rst
    ├── tests
    │   ├── requirements.txt
    │   ├── integration_test.py
    │   └── benchmark
    │       ├── requirements.txt
    │       ├── main.py
    │       ├── rosenbrock.py
    │       ├── {algoname}.yaml
    │       ├── bayesopt.yaml
    │       └── random_search.yaml
    └── src
        └── orion
            └── algo
                └── {plugin_name}
                    ├── {algoname}.py
                    ├── __init__.py
                    └── _version.py

The important files to modify are ``src/orion/algo/{plugin_name}/{module_name}.py`` to implement the
algorithm and ``tests/benchmark/{algo_name}.yaml`` to fill the arguments required for the algorithm
you implement.

``LICENSE``

Note that you are free to change the License, copyright is to your name.

``versioneer.py``
``src/orion/algo/{plugin_name}/_version.py``

This serves to version automatically your algo, just ignore these if you don't plan to make
releases.

``tests``

These are the automatically generated tests.

``tests/benchmark``

Automatically generated benchmark test using the yaml files created in the same folder.

``doc``

Automatically generated template for documentation

``tox.ini``

Tox file defining commands to run tests, build doc and publish code.

Implementation
==============

.. code-block:: python

    requires = 'real'

Some algorithms requires the search space to be real. You can specify this requirement by adding the
attribute ``requires = 'real'`` at the base of the class definition.In this case, the algorithm
wrapper in Orion's core will convert the search space to real one before passing it to your
algorithm. This way the user can define discrete or categorital dimensions while using algorithms
that require a real space.

.. code-block:: python

    def __init__(self, space, seed=None):

The initialization of the algorithm must pass space and seed to ``super().__init__``, but must also
pass any other argument that must be part of the configuration of the algorithm. Any argument passed
to ``super()`` will be assigned as an attribute to the algorithm and will be included in
``algo.configuration``, which is used to save the configuration of the algorithm in the database.


.. code-block:: python

    def seed_rng(self, seed=None):

This method must seed the internal state of the algorithm so that it would always sample the same
sequence of points.

.. code-block:: python

    @property
    def state_dict(self):

The state dict is used to copy algorithms within the parallel strategy. All algorithms must provide
a state dict to ensure that we reset it to a previous state.

.. code-block:: python

    def set_state(self, state_dict):

Stateful attributes of the algorithm are reset using the given ``state_dict``. Note that
``set_state`` must be compliant with ``state_dict`` and use
the same structure.

.. code-block:: python

    def suggest(self, num=1):

The method to suggest new trials. The argument ``num=1``
request the number of trials that the algorithm must sample. Note that it is possible to only
support ``num=1`` and raise ValueError otherwise.

.. code-block:: python

    def observe(self, points, results):

The method to observe results of suggested trials. Note that observe may be called several times for
the same points. Make sure to handle this properly within your algorithm if this is problematic.
Points are passed as a list of lists, each list representing the value of the params in the order
defined in ``self.space``

Tests
=====

To test the freshly built package, you must first install the requirements. From within the new
package, run

.. code-block:: console

    $ pip install -r tests/requirements.txt

You can then run the unit-tests with

.. code-block:: console

    $ pytest tests/integration_test.py

or using ``tox``

.. code-block:: console

    $ tox -e py36

Note that the algorithm pre-built is random search so that you can start from a fully working
environment and test your way through the modifications.

There is also the option of running the toy-benchmark to compare the performance of your algorithm
with random search and bayesian optimization. First install the requirements.

.. code-block:: console

    $ pip install -r tests/benchmark/requirements.txt

And then execute the benchmark

.. code-block:: console

    $ pytest tests/benchmark/main.py

or using ``tox``

.. code-block:: console

    $ tox -e benchmark

Finally, official plugins must follow the same code quality standards than ``orion.core``. Therefore
there is tests included in the pre-built package for ``flake8`` and ``pylint``. You can execute them
with

.. code-block:: console

    $ tox -e flake8

and

.. code-block:: console

    $ tox -e pylint
