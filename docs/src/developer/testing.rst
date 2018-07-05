.. contents:: Developer's Guide 101: Tools & Testing

*******
Testing
*******

For developer's convenience the packages enlisted in the requirements file
``dev-requirements.txt`` are meant to facilitate the development process.
Packages include `tox <https://tox.readthedocs.io/en/latest/>`_ for defining
and organizing macros of sh commands in virtual environments, and packages
for linting as we will see in a next chapter.


Continuous Integration
======================

We use **TravisCI** and **CodeCov**.

.. image:: https://travis-ci.org/mila-udem/orion.svg?branch=master
   :target: https://travis-ci.org/mila-udem/orion

.. image:: https://codecov.io/gh/mila-udem/orion/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mila-udem/orion

Continuous Testing
==================

Using ``tox`` we can automate many processes of continuous testing into macros.
All contexts are defined in `/tox.ini <https://github.com/mila-udem/orion/blob/master/tox.ini>`_.

By calling::

   tox

one attempts to call all contexts that matter for our Continuous Integration in
the same call. Those are *py34*, *py35*, *py36*, *py37* for running tests and
checking coverage, *flake8*, *pylint*, *doc8*, *packaging* for linting code,
documentation and Python packaging-related files, and finally *docs* for
building the Sphinx documentation.

.. code-block:: sh

   tox -e py

This will call tests for the particular shell's environment Python's executable.
If the tests are successful, then a code **coverage** summary will be printed
on shell's screen.

.. code-block:: sh

   tox -e devel

This will finally always run the tests on background and on a code change event,
it automatically performs **regression testing**.

Test
====

All the tests for our software are located and organized in the directory
``/tests`` relative to the root of the code repository. There are two kinds of
tests: Unit tests are located under ``/tests/unittests`` and functional tests
(tests which invoke Oríon's executable from shell) under ``/tests/functional``.

Our software requires pytest_ ``>=3.0.0`` for automated testing.
Also, it requires the particular database setup described in
:doc:`/database` to have been followed.

Hence the tests can be invoked with::

   python setup.py test

For instance::

   python setup.py test --addopts 'tests/unittests'

will only execute tests located under ``/tests/unittests``, this is all unit
tests.

.. _pytest: https://docs.pytest.org/en/latest/
