*******
Testing
*******

All the tests for our software are located and organized in the directory
``/tests`` relative to the root of the code repository. There are three kinds of
tests:

#. Unit tests, located under ``/tests/unittests``.
   They test individual features, often at the class granularity.
#. Functional tests, located under ``/tests/functional``.
   They test end to end functionality, invoking Oríon's executable from shell.
#. Stress tests, located under ``/tests/stress``.
   They test the resilience and performance.

The tests are made with pytest_. We highly recommend you check it out and take a look at the
existing tests in the ``tests`` folder.

We recommend invoking the tests using ``tox`` as this will be the method used by the CI system.
It will avoid you headaches when trying to run tests and nasty surprises when submitting PRs.

.. warning::

   MongoDB is required to be installed and running as tests depend on it. If MongoDB is not
   installed, please follow these :ref:`installation instructions <mongodb_install>` first.

Running tests
=============
To run the complete test suite, you can use

.. code-block:: sh

   $ tox -e py

This will call tests for the particular shell's environment Python's executable.
If the tests are successful, then a code **coverage** summary will be printed
on shell's screen.

However, during development consider using

.. code-block:: sh

   $ tox -e devel

This will run in the background and run the tests on a code change event (e.g., you save a file)
automatically run the tests when you make a change. It's particularly useful when you also
specify the location of the tests:

.. code-block:: sh

   $ tox -e devel -- 'path/to/your/tests/'

.. code-block:: sh

   $ tox -e devel -- 'path/to/your/tests/file.py'

.. code-block:: sh

   $ tox -e devel -- 'path/to/your/tests/file.py::test_name'

This way, the tests will be ran automatically every time you make a change in the specified folder,
file, or test respectively. This option is also available for ``$ tox -e py``.

.. _pytest: https://docs.pytest.org/en/latest/


.. include:: stress.rst
