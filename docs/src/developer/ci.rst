**********************
Continuous Integration
**********************
.. image:: https://travis-ci.org/epistimio/orion.svg?branch=master
   :target: https://travis-ci.org/epistimio/orion

.. image:: https://codecov.io/gh/epistimio/orion/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/epistimio/orion

We use travis-ci_ and codecov_ for continuous integration and tox_ to automate the process at
the repository level.

When a commit is pushed in a pull request, a call to ``$ tox`` is made by
TravisCI which triggers the following chain of events:

#. A test environment is spun up for each version of python tested (definined in ``tox.ini``).
#. Code styles verifications, and quality checks are run (``flake8``, ``pylint``, ``doc8``). The
   documentation is also built at this time (``docs``).
#. The test suite is run completely with coverage, including the dedicated backward
   compatibility tests.
#. The structure of the repository is validated by ``check-manifest`` and ``readme_renderer``.
#. The results of the coverage check are reported directly in the pull request.

The coverage results show the difference of coverage introduced by the changes. We always aim to
have changes that improve coverage.

If a step fails at any point in any environment, the build will be immediatly stopped, marked as
failed and reported to the pull requestion and repository. In such case, the maintainers and
relevant contributors will be alerted.

.. _codecov: https://codecov.io/
.. _travis-ci: https://travis-ci.com/
.. _tox: https://tox.readthedocs.io/en/latest/
