**********************
Continuous Integration
**********************
.. image:: https://travis-ci.org/epistimio/orion.svg?branch=master
   :target: https://travis-ci.org/epistimio/orion

.. image:: https://codecov.io/gh/epistimio/orion/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/epistimio/orion

We use travis-ci_ and codecov_ for continuous integration and tox_ to automate the process at
the repository level. When a commit is pushed in a pull request, a call to ``$ tox`` will be made by
TravisCI which will trigger a call to multiple contexts. *py35*, *py36*, *py37*, ... for running
tests and checking coverage, *flake8*, *pylint*, *doc8*, *packaging* for linting code,
documentation and Python packaging-related files, and finally *docs* for
building the Sphinx documentation.

.. _codecov: https://codecov.io/
.. _travis-ci: https://travis-ci.com/
.. _tox: https://tox.readthedocs.io/en/latest/
