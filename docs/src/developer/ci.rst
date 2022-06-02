.. _ci:

**********************
Continuous Integration
**********************
.. image:: https://github.com/Epistimio/orion/workflows/build/badge.svg?branch=master&event=pull_request
    :target: https://github.com/Epistimio/orion/actions?query=workflow:build+branch:master+event:schedule
    :alt: Github actions tests

.. image:: https://codecov.io/gh/epistimio/orion/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/epistimio/orion

We use github-actions_ and codecov_ for continuous integration and tox_ to automate the process at
the repository level.

When a commit is pushed in a pull request, a github workflow is spawned which
triggers the following chain of events:

#. Code styles verifications, and quality checks are run
   (``black``, ``isort``, ``pylint``, ``doc8``).
   The documentation is also built at this time (``docs``).
#. When code style verifications and documentation built passes, a test environment is spun up for
   each version of python tested (defined in ``.github/workflows/build.yml``).
#. The test suite is run completely with coverage, including the dedicated backward
   compatibility tests.
#. When all code tests passes, the structure of the repository is validated by ``check-manifest``
   and ``readme_renderer``, and packaging for PyPi and Conda is tested.

The coverage results show the difference of coverage introduced by the changes. We always aim to
have changes that improve coverage.

If a step fails at any point in any environment, the build will be immediately stopped, marked as
failed and reported to the pull request and repository. In such case, the maintainers and
relevant contributors will be alerted.


.. tip::

   We recommend using `pre-commit`_ to validate your changes locally before they get committed.

   This greatly reduces the turnaround time when working on a pull request by avoiding builds
   failing due to the initial filters of the CI loop (black, isort, pylint, doc8, etc). It also
   helps you to write modern python code by avoiding older syntax.

   To get started with pre-commit, simply install and enable it like so:

   .. code-block:: sh

      $ pip install pre-commit
      $ pre-commit install


The workflow described above is also executed daily to detect any break due to change in
dependencies. When releases are made, the workflow is also executed and additionally
publish the release to PyPi_ and Conda_.

.. _codecov: https://codecov.io/
.. _github-actions: https://docs.github.com/en/free-pro-team@latest/actions
.. _tox: https://tox.readthedocs.io/en/latest/
.. _PyPI: https://pypi.org/project/orion/
.. _Conda: https://anaconda.org/epistimio/orion
.. _pre-commit: https://pre-commit.com/
